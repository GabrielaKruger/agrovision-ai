import os
import cv2

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Importações dos Services
from services.config import settings
from services.schemas import ChatRequest, ChatResponse
from services.event_repository import init_db, list_events
from services.video_monitor import start_monitoring_thread, get_latest_frame
from services.monitoring_agent import AGENT_PROFILE, build_agent_messages, build_event_context
from services.ollama_client import chat_with_ollama
from services.scraping_service import fetch_market_data

# =========================
# INICIALIZAÇÃO DO APP E FASTAPI
# =========================
app = FastAPI(title="AgroVision AI")

# Pastas necessárias
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs(settings.SAVE_DIR, exist_ok=True)

# Montagem Estática e Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Threads & Banco
@app.on_event("startup")
def startup_event():
    init_db()
    start_monitoring_thread()
    print("[INFO] AgroVision AI iniciado através das novas camadas modulares.")

# =========================
# ROTAS FRONT-END
# =========================
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    """Página principal com eventos e chat."""
    events = list_events(20)
    return templates.TemplateResponse("index.html", {"request": request, "events": events})

@app.get("/frame")
def get_frame():
    """Retorna o frame global atual do loop de vídeo."""
    frame = get_latest_frame()
    if frame is None:
        return JSONResponse(status_code=503, content={"message": "Ainda sem frame."})
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return JSONResponse(status_code=500, content={"message": "Erro ao encodar frame."})
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

# =========================
# ROTAS DA API REST COM OS AGENTES (Evolução)
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "service": "AgroVision AI API"}

@app.get("/events")
def get_events():
    return JSONResponse(content=list_events(50))

@app.get("/market")
def get_market_info():
    """Retorna os dados de cotações agrícolas coletados via web scraping."""
    return JSONResponse(content=fetch_market_data())

@app.get("/camera/status")
def camera_status():
    frame = get_latest_frame()
    return {
        "online": frame is not None,
        "source": settings.CAMERA_SOURCE,
        "has_live_frame": frame is not None,
    }

@app.get("/agent/status")
def agent_status():
    """Visualiza contexto e perfis do Agente atual."""
    recent = list_events(settings.AGENT_EVENT_LIMIT)
    return {
        "name": AGENT_PROFILE.name,
        "role": AGENT_PROFILE.role,
        "goal": AGENT_PROFILE.goal,
        "events_in_context": len(recent),
        "context_preview": build_event_context(recent)
    }

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    """Rota para interagir com o modelo e solicitar avaliações do Agente."""
    question = payload.message
    history = [] # Simulação. A sessão de histórico poderia vir do cliente no futuro.
    
    events = list_events(settings.AGENT_EVENT_LIMIT)
    
    # 1. Monta as mensagens estruturadas (system prompt, context, e user text)
    messages = build_agent_messages(question, history, events)
    
    # 2. Chama a LLM
    answer = chat_with_ollama(messages)
    
    return ChatResponse(answer=answer)