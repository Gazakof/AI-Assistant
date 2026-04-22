import sqlite3
import os
from flask import current_app

def get_db_path():
    # 📂 chemin vers le dossier instance de Flask
    instance_path = current_app.instance_path

    # créer le dossier si nécessaire
    os.makedirs(instance_path, exist_ok=True)

    return os.path.join(instance_path, "chat_memory.db")


def init_db():
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT
    )
    """)

    conn.commit()
    conn.close()


def save_memory(question, answer):
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO memory (question, answer) VALUES (?, ?)",
        (question, answer)
    )

    conn.commit()
    conn.close()