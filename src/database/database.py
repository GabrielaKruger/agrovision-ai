import sqlite3
from datetime import datetime

DB_PATH = "detections.db"

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