import os
import cv2
import time
import uuid
import sqlite3
import threading
from datetime import datetime
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

CAMERA_SOURCE = 0
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.45
SAVE_DIR = "static/captures"
DB_PATH = "detections.db"

TARGET_CLASSES = {"person", "car", "motorcycle", "truck", "bus"}

app = FastAPI()

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = YOLO(MODEL_PATH)

last_frame = None
last_frame_lock = threading.Lock()

detection_state = defaultdict(int)
last_alert_time = defaultdict(lambda: 0.0)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            event_time TEXT,
            label TEXT,
            confidence REAL,
            image_path TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_event(event_id, label, confidence, image_path):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO events VALUES (?, ?, ?, ?, ?)
    """, (
        event_id,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        label,
        confidence,
        image_path
    ))
    conn.commit()
    conn.close()

def list_events(limit=20):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM events
        ORDER BY event_time DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows

def process_stream():
    global last_frame

    cap = cv2.VideoCapture(CAMERA_SOURCE)

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(1)
            continue

        results = model(frame, conf=CONFIDENCE_THRESHOLD)

        for result in results:
            for box in result.boxes:
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])

                if label in TARGET_CLASSES:
                    event_id = str(uuid.uuid4())[:8]
                    filename = f"{event_id}.jpg"
                    filepath = os.path.join(SAVE_DIR, filename)

                    cv2.imwrite(filepath, frame)
                    save_event(event_id, label, conf, filepath)

        with last_frame_lock:
            last_frame = frame.copy()

        time.sleep(0.1)

@app.on_event("startup")
def startup():
    init_db()
    thread = threading.Thread(target=process_stream, daemon=True)
    thread.start()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/events")
def events():
    return list_events()

@app.get("/frame")
def frame():
    global last_frame
    with last_frame_lock:
        if last_frame is None:
            return {"msg": "sem frame"}
        _, buffer = cv2.imencode(".jpg", last_frame)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")