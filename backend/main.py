import asyncio
from utils import (
    record_audio,
    speech_to_text,
    get_ai_response,
    make_spoken_reply,
    text_to_speech
)
import logging
from config import config

async def voice_chat():
    print("\n🎙️ Advanced VoiceAI Assistant")
    print("============================")
    print("Press Ctrl+C to exit\n")
    
    while True:
        try:
            # Record user audio
            audio_file = record_audio()
            if not audio_file:
                print("⚠️ Recording failed. Try again.")
                continue

            # Convert to text
            user_text = speech_to_text(audio_file)
            if not user_text:
                print("⚠️ No speech detected. Try again.")
                continue

            print(f"\n👤 You: {user_text}")
            
            # Get AI response
            ai_response = get_ai_response(user_text)
            print(f"🤖 AI: {ai_response}")
            
            # Prepare and speak response
            reply = make_spoken_reply(ai_response)
            text_to_speech(reply)

        except KeyboardInterrupt:
            print("\n👋 Exiting VoiceAI. Goodbye!")
            break
        except Exception as e:
            logging.error(f"Main loop error: {e}")
            print(f"❌ An error occurred: {e}")
            await asyncio.sleep(1)  # Prevent tight error loop

if __name__ == "__main__":
    try:
        asyncio.run(voice_chat())
    except Exception as e:
        logging.critical(f"Application crash: {e}")
        print(f"💥 Critical error: {e}")