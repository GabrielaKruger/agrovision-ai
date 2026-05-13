import os
import cv2
import time
import uuid
import threading
from datetime import datetime
from collections import defaultdict

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

# Import our modules
from src.database.database import init_db, save_event, list_events
from src.services.ai_service import detect_objects
from src.services.scraping_service import get_weather_data, get_agricultural_news, get_commodity_prices

# Configuration
CAMERA_SOURCE = 0
SAVE_DIR = "static/captures"

# Security
security = HTTPBasic()
USERNAME = "admin"
PASSWORD = "password"  # In production, use environment variables

TARGET_CLASSES = {"person", "car", "motorcycle", "truck", "bus"}

app = FastAPI()

os.makedirs("static", exist_ok=True)
os.makedirs("src/frontend/templates", exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="src/frontend/templates")

last_frame = None
last_frame_lock = threading.Lock()

detection_state = defaultdict(int)
last_alert_time = defaultdict(lambda: 0.0)

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return credentials.username

def process_stream():
    global last_frame

    cap = cv2.VideoCapture(CAMERA_SOURCE)

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(1)
            continue

        # Use AI service for detection
        detections = detect_objects(frame)

        for detection in detections:
            event_id = str(uuid.uuid4())[:8]
            filename = f"{event_id}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)

            cv2.imwrite(filepath, frame)
            save_event(event_id, detection['label'], detection['confidence'], filepath)

        with last_frame_lock:
            last_frame = frame.copy()

        time.sleep(0.1)

@app.on_event("startup")
def startup():
    init_db()
    thread = threading.Thread(target=process_stream, daemon=True)
    thread.start()

@app.get("/", response_class=HTMLResponse)
def home(request: Request, username: str = Depends(authenticate)):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/events")
def events(username: str = Depends(authenticate)):
    try:
        return list_events()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/frame")
def frame(username: str = Depends(authenticate)):
    global last_frame
    with last_frame_lock:
        if last_frame is None:
            return {"msg": "sem frame"}
        _, buffer = cv2.imencode(".jpg", last_frame)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

@app.get("/weather")
def weather(location: str = "São Paulo, Brazil", username: str = Depends(authenticate)):
    try:
        data = get_weather_data(location)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Scraping error")

@app.get("/news")
def news(username: str = Depends(authenticate)):
    try:
        data = get_agricultural_news()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Scraping error")

@app.get("/commodities")
def commodities(username: str = Depends(authenticate)):
    try:
        data = get_commodity_prices()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Scraping error")

@app.get("/captures")
def captures(username: str = Depends(authenticate)):
    try:
        import os
        captures_dir = "src/frontend/static/captures"
        if os.path.exists(captures_dir):
            files = [f for f in os.listdir(captures_dir) if f.endswith('.jpg')]
            return {"captures": files}
        return {"captures": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error listing captures")