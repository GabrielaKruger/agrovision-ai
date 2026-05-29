import os
from dotenv import load_dotenv

# Carrega do .env
load_dotenv()

class Settings:
    # Câmera e Modelo YOLO
    CAMERA_SOURCE: str = os.getenv("CAMERA_SOURCE", "0")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "yolo11n.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
    MIN_CONSECUTIVE_FRAMES: int = int(os.getenv("MIN_CONSECUTIVE_FRAMES", "3"))
    ALERT_COOLDOWN_SECONDS: int = int(os.getenv("ALERT_COOLDOWN_SECONDS", "20"))
    
    # Paths locais
    SAVE_DIR: str = os.getenv("SAVE_DIR", "static/captures")
    DB_PATH: str = os.getenv("DB_PATH", "detections.db")
    
    # Configurações do Ollama
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
    AGENT_EVENT_LIMIT: int = int(os.getenv("AGENT_EVENT_LIMIT", "12"))

# Instância global das configurações
settings = Settings()

# Classes Alvo
TARGET_CLASSES = {"person", "carro", "motocicleta", "onibus"}