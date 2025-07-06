import os
from dotenv import load_dotenv
from pathlib import Path
from dataclasses import dataclass, field
import pyaudio

load_dotenv()

@dataclass
class AudioConfig:
    SAMPLE_RATE: int = 16000
    CHUNK_SIZE: int = 1024
    FORMAT: int = pyaudio.paInt16
    CHANNELS: int = 1
    SILENCE_THRESHOLD: int = 800
    SILENCE_DURATION: float = 1.8
    INPUT_DEVICE_INDEX: int = None

@dataclass
class TTSConfig:
    VOICE_ID: str = "SAz9YHcvj6GT2YYXdXww"
    MODEL_ID: str = "eleven_multilingual_v2"
    FALLBACK_VOICE: bool = True

@dataclass
class STTConfig:
    MODEL: str = "base"
    LANGUAGE: str = "en"
    TEMPERATURE: float = 0.0

@dataclass
class AIConfig:
    MODEL: str = "mistralai/mistral-nemotron"
    API_BASE: str = "https://integrate.api.nvidia.com/v1"
    MAX_TOKENS: int = 200
    TEMPERATURE: float = 0.7
    SYSTEM_PROMPT: str = (
        "You are a friendly voice assistant. "
        "Give concise yet complete, human-like answers."
    )

@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    LOG_DIR: str = "logs"
    CACHE_DIR: str = "cache"

config = AppConfig()

# Create necessary directories
Path(config.LOG_DIR).mkdir(exist_ok=True)
Path(config.CACHE_DIR).mkdir(exist_ok=True)