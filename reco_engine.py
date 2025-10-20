import json, os, random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "recommendations.json")

class RecoEngine:
    def __init__(self, path=DATA_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró {path}")
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def get(self, emotion: str, k: int | None = None):

        emo = emotion if emotion in self.data else "neutral"
        pack = self.data.get(emo, {})
        out = {}
        for key in ("music", "movies", "activities"):
            items = list(pack.get(key, []))
            random.shuffle(items)
            out[key] = items if k is None else items[:k]
        return out
