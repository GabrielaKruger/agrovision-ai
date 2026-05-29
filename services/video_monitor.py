import os
import cv2
import time
import uuid
import threading
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

from .config import settings, TARGET_CLASSES
from .event_repository import save_event

# Frame compartilhado entre a thread de leitura e as rotas
last_frame = None
last_frame_lock = threading.Lock()

# Estado de detecção por classe
detection_state = defaultdict(int)
last_alert_time = defaultdict(lambda: 0.0)

try:
    model = YOLO(settings.MODEL_PATH)
except Exception as e:
    print(f"[AVISO] Não foi possível carregar o modelo YOLO: {e}")
    model = None

def get_latest_frame():
    """Retorna o frame global mais recente protegido por lock."""
    with last_frame_lock:
        if last_frame is None:
            return None
        return last_frame.copy()

def draw_box(frame, x1: int, y1: int, x2: int, y2: int, label: str, conf: float):
    """Desenha o bounding box e o rótulo sobre o frame."""
    text = f"{label} {conf:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        text,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

def should_alert(label: str) -> bool:
    """Verifica se o cooldown do alerta para a classe já expirou."""
    now = time.time()
    return (now - last_alert_time[label]) > settings.ALERT_COOLDOWN_SECONDS

def process_stream():
    """Loop principal de captura e detecção."""
    global last_frame

    if str(settings.CAMERA_SOURCE).isdigit():
        camera = int(settings.CAMERA_SOURCE)
    else:
        camera = settings.CAMERA_SOURCE

    while True:
        cap = cv2.VideoCapture(camera)
        
        if not cap.isOpened():
            print(f"[ERRO] Falha ao abrir a fonte de vídeo {camera}. Tentando novamente em 5s...")
            time.sleep(5)
            continue
            
        print("[INFO] Fonte de vídeo aberta com sucesso.")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[AVISO] Frame não lido. Tentando reconectar...")
                time.sleep(1)
                break # Sai do loop para reiniciar cap

            if model is None:
                # Fallback caso dê algum erro de import, apenas mostra a camera
                with last_frame_lock:
                    last_frame = frame.copy()
                time.sleep(0.05)
                continue

            # Inferência do YOLO
            results = model(frame, conf=settings.CONFIDENCE_THRESHOLD, verbose=False)

            found_labels_in_frame: set = set()
            best_conf_by_label: dict = {}

            for result in results:
                boxes = result.boxes
                if boxes is None: continue

                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    label = model.names[cls_id]

                    if label not in TARGET_CLASSES:
                        continue

                    found_labels_in_frame.add(label)

                    if label not in best_conf_by_label or conf > best_conf_by_label[label]:
                        best_conf_by_label[label] = conf

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    draw_box(frame, x1, y1, x2, y2, label, conf)

            # Atualiza o estado das detecções consecutivas
            for label in TARGET_CLASSES:
                if label in found_labels_in_frame:
                    detection_state[label] += 1
                else:
                    detection_state[label] = 0

            # Dispara alerta e salva
            for label in found_labels_in_frame:
                if detection_state[label] >= settings.MIN_CONSECUTIVE_FRAMES and should_alert(label):
                    event_id = str(uuid.uuid4())[:8]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{timestamp}_{label}_{event_id}.jpg"
                    filepath = os.path.join(settings.SAVE_DIR, filename)

                    cv2.imwrite(filepath, frame)
                    image_path = f"/static/captures/{filename}"

                    confidence = best_conf_by_label.get(label, 0.0)
                    save_event(event_id, label, confidence, image_path)

                    last_alert_time[label] = time.time()
                    print(f"[ALERTA] {label} detectado! Evidência em: {filepath}")

            # Atualiza frame global
            with last_frame_lock:
                last_frame = frame.copy()

            time.sleep(0.05)
        
        cap.release()

def start_monitoring_thread():
    """Inicia a thread de captura de vídeo."""
    thread = threading.Thread(target=process_stream, daemon=True)
    thread.start()