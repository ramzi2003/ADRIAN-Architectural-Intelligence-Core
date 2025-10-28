"""
Shared configuration management for ADRIAN services.
Loads environment variables and provides configuration objects.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    """Global settings for all ADRIAN services."""
    
    # Application
    app_name: str = "ADRIAN"
    environment: str = "development"
    debug: bool = True
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "adrian"
    postgres_user: str = "adrian"
    postgres_password: str = "adrian_password"
    
    # Ollama (Local LLM)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "mistral:7b"
    
    # Gemini (Cloud LLM)
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-pro"
    
    # Porcupine (Hotword Detection)
    picovoice_access_key: Optional[str] = None
    hotword_sensitivity: float = 0.7  # 0.1-1.0
    
    # Speech-to-Text (Whisper)
    whisper_model_name: str = "base"  # "tiny", "base", "small", "medium", "large"
    whisper_device: str = "cpu"  # "cpu" or "cuda"
    stt_confidence_threshold: float = 0.5  # 0.0-1.0
    
    # Conversation Manager
    conversation_timeout_seconds: float = 30.0  # Seconds of silence before returning to idle
    max_context_turns: int = 3  # Number of previous conversation turns to remember
    
    # Voice Activity Detection (VAD)
    vad_aggressiveness: int = 1  # 0-3 (0=least aggressive, 3=most aggressive)
    
    # Text-to-Speech (Coqui TTS)
    # Using fast VITS model with selected British male voice (p234)
    tts_model_name: str = "tts_models/en/vctk/vits"  # Multi-speaker VITS (fast model, already downloaded)
    tts_speaker: str = "p234"  # Chosen default voice
    tts_use_voice_cloning: bool = False  # Disable XTTS cloning, use VCTK voices
    tts_speed: float = 1.0  # Normal pace for clarity
    tts_pitch_shift: float = -2.5  # Deeper, authoritative tone
    tts_device: str = "cpu"  # "cpu" or "cuda"
    
    # Service Ports
    io_service_port: int = 8001
    processing_service_port: int = 8002
    memory_service_port: int = 8003
    execution_service_port: int = 8004
    security_service_port: int = 8005
    output_service_port: int = 8006
    
    # Logging
    log_level: str = "INFO"
    log_db_path: str = "logs/adrian.db"
    
    # FAISS
    faiss_index_path: str = "data/faiss_index"
    embedding_dimension: int = 384
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    # Find .env file in project root
    current_dir = Path.cwd()
    
    # Look for .env file in current directory and parent directories
    for _ in range(5):  # Look up to 5 levels up
        env_path = current_dir / ".env"
        if env_path.exists():
            # Set the working directory to where .env is found
            os.environ['PWD'] = str(current_dir)
            break
        current_dir = current_dir.parent
        if current_dir == current_dir.parent:  # Reached root
            break
    
    return Settings()


# Database URLs
def get_postgres_url(settings: Optional[Settings] = None) -> str:
    """Generate PostgreSQL connection URL."""
    if settings is None:
        settings = get_settings()
    return (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )


def get_postgres_async_url(settings: Optional[Settings] = None) -> str:
    """Generate async PostgreSQL connection URL."""
    if settings is None:
        settings = get_settings()
    return (
        f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )

