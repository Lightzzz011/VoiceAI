import os
import wave
import time
import audioop
import threading
import numpy as np
import pyaudio
import whisper
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from simpleaudio import WaveObject
from pydub import AudioSegment
from io import BytesIO
import logging
from datetime import datetime
from config import config

# Initialize logging
logging.basicConfig(
    filename=f"{config.LOG_DIR}/voiceai_{datetime.now().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize models and clients
whisper_model = whisper.load_model(config.stt.MODEL)
eleven = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Global variables for TTS control
stop_speaking = False
current_play_obj = None
conversation_history = []

def record_audio(filename="user_input.wav"):
    """Record audio with silence detection and auto-stop"""
    p = pyaudio.PyAudio()
    
    stream = p.open(
        format=config.audio.FORMAT,
        channels=config.audio.CHANNELS,
        rate=config.audio.SAMPLE_RATE,
        input=True,
        frames_per_buffer=config.audio.CHUNK_SIZE,
        input_device_index=config.audio.INPUT_DEVICE_INDEX
    )

    logging.info("Recording started")
    print("\n🎤 Listening... Speak now.")
    frames = []
    speaking = False
    silence_start = None

    try:
        while True:
            data = stream.read(config.audio.CHUNK_SIZE, exception_on_overflow=False)
            rms = audioop.rms(data, 2)
            
            if rms >= config.audio.SILENCE_THRESHOLD:
                if not speaking:
                    print("🎙️ Detected speech...")
                    speaking = True
                frames.append(data)
                silence_start = None
            elif speaking:
                frames.append(data)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > config.audio.SILENCE_DURATION:
                    print("🛑 Silence detected. Stopping recording.")
                    break

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(config.audio.CHANNELS)
        wf.setsampwidth(p.get_sample_size(config.audio.FORMAT))
        wf.setframerate(config.audio.SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    return optimize_audio(filename)

def optimize_audio(audio_file):
    """Enhance audio quality for better STT accuracy"""
    try:
        audio = AudioSegment.from_wav(audio_file)
        audio = audio.normalize().high_pass_filter(100).low_pass_filter(4000)
        optimized_file = f"{config.CACHE_DIR}/optimized_input.wav"
        audio.export(optimized_file, format="wav")
        return optimized_file
    except Exception as e:
        logging.error(f"Audio optimization failed: {e}")
        return audio_file

def speech_to_text(audio_file):
    """Convert speech to text using Whisper"""
    try:
        result = whisper_model.transcribe(
            audio_file,
            language=config.stt.LANGUAGE,
            temperature=config.stt.TEMPERATURE
        )
        return result["text"].strip()
    except Exception as e:
        logging.error(f"STT failed: {e}")
        return ""

def get_ai_response(user_input):
    """Generate AI response with conversation context"""
    global conversation_history
    
    try:
        client = OpenAI(
            base_url=config.ai.API_BASE,
            api_key=os.getenv("NVIDIA_API_KEY")
        )
        
        messages = [{"role":"system","content":config.ai.SYSTEM_PROMPT}]
        messages.extend(conversation_history[-6:])  # Keep last 3 exchanges
        messages.append({"role":"user","content":user_input})
        
        resp = client.chat.completions.create(
            model=config.ai.MODEL,
            messages=messages,
            temperature=config.ai.TEMPERATURE,
            max_tokens=config.ai.MAX_TOKENS
        )
        
        ai_response = resp.choices[0].message.content.strip()
        
        # Update conversation history
        conversation_history.extend([
            {"role":"user","content":user_input},
            {"role":"assistant","content":ai_response}
        ])
        
        return ai_response
        
    except Exception as e:
        logging.error(f"AI response failed: {e}")
        return get_fallback_response(user_input)

def get_fallback_response(user_input):
    """Local fallback when cloud APIs fail"""
    # Simple rule-based responses
    if any(word in user_input.lower() for word in ["hello", "hi", "hey"]):
        return "Hello! I'm having trouble with my main system, but I can still help."
    return "I'm sorry, I'm experiencing technical difficulties. Please try again later."

def make_spoken_reply(ai_text):
    """Prepare text for natural-sounding speech"""
    sentences = [s.strip() for s in ai_text.split(".") if s.strip()]
    if len(sentences) > 2:
        return ". ".join(sentences[:2]) + "."
    return ai_text

def listen_for_interrupt():
    """Detect user speech to interrupt TTS"""
    global stop_speaking
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=config.audio.FORMAT,
        channels=config.audio.CHANNELS,
        rate=config.audio.SAMPLE_RATE,
        input=True,
        frames_per_buffer=config.audio.CHUNK_SIZE
    )
    
    start_time = None
    try:
        while not stop_speaking:
            data = stream.read(config.audio.CHUNK_SIZE, exception_on_overflow=False)
            volume = np.linalg.norm(np.frombuffer(data, dtype=np.int16))
            
            if volume > config.audio.SILENCE_THRESHOLD * 3:  # Higher threshold for interrupts
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= 0.5:  # 500ms of continuous speech
                    stop_speaking = True
                    break
            else:
                start_time = None
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def text_to_speech(text, emotion="neutral"):
    """Convert text to speech with interrupt capability"""
    global stop_speaking, current_play_obj
    
    stop_speaking = False
    current_play_obj = None
    
    def _play():
        global current_play_obj
        try:
            # ElevenLabs TTS
            audio_chunks = eleven.text_to_speech.convert(
                voice_id=config.tts.VOICE_ID,
                text=text,
                model_id=config.tts.MODEL_ID
            )
            audio_bytes = b"".join(audio_chunks)
            
            # Process audio
            mp3_buf = BytesIO(audio_bytes)
            audio_seg = AudioSegment.from_file(mp3_buf, format="mp3")
            raw = audio_seg.raw_data
            
            # Play audio
            wave_obj = WaveObject(
                raw, 
                audio_seg.channels,
                audio_seg.sample_width,
                audio_seg.frame_rate
            )
            play_obj = wave_obj.play()
            current_play_obj = play_obj
            play_obj.wait_done()
            
        except Exception as e:
            logging.error(f"TTS failed: {e}")
            if config.tts.FALLBACK_VOICE:
                _fallback_tts(text)

    def _fallback_tts(text):
        """System TTS fallback"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logging.error(f"Fallback TTS failed: {e}")
            print(f"🤖 AI: {text}")  # Final fallback

    # Start threads
    tts_thread = threading.Thread(target=_play, daemon=True)
    intr_thread = threading.Thread(target=listen_for_interrupt, daemon=True)
    
    tts_thread.start()
    time.sleep(1.0)  # Delay to avoid catching speaker output
    intr_thread.start()

    # Monitor playback
    while tts_thread.is_alive():
        if stop_speaking and current_play_obj:
            print("\n🛑 Interrupted—stopping TTS.")
            current_play_obj.stop()
            break
        time.sleep(0.05)