import os
import json
import numpy as np
from typing import Dict, Any

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
USERS_PATH = os.path.join(DATA_DIR, "users.json")

DEFAULT_DB: Dict[str, Any] = {"users": {}}


def _ensure_dirs_and_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(USERS_PATH) or os.path.getsize(USERS_PATH) == 0:
        # crea o repara archivo vacío
        with open(USERS_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_DB, f, ensure_ascii=False, indent=2)


def _safe_read_json(path: str) -> Dict[str, Any]:
    _ensure_dirs_and_file()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "users" not in data or not isinstance(data["users"], dict):
            raise ValueError("Estructura inválida")
        return data
    except Exception:
        # Repara el archivo corrupto
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_DB, f, ensure_ascii=False, indent=2)
        return DEFAULT_DB.copy()

def load_users() -> Dict[str, Any]:
    return _safe_read_json(USERS_PATH)

def save_users(data: Dict[str, Any]):
    os.makedirs(DATA_DIR, exist_ok=True)
    # escritura "atómica" simple
    tmp_path = USERS_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, USERS_PATH)

def get_user_embedding(username: str):
    db = load_users()
    u = db.get("users", {}).get(username)
    if not u:
        return None
    vec = u.get("embedding")
    if not vec:
        return None
    return np.array(vec, dtype=np.float32)

def set_user_embedding(username: str, embedding):
    db = load_users()
    if "users" not in db or not isinstance(db["users"], dict):
        db = DEFAULT_DB.copy()
    db["users"][username] = {"embedding": np.asarray(embedding, dtype=float).tolist()}
    save_users(db)

def user_exists(username: str) -> bool:
    db = load_users()
    return isinstance(db, dict) and "users" in db and username in db["users"]
