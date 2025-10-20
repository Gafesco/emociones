import numpy as np
from deepface import DeepFace

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-8
    return v / n

def face_embedding(img_bgr, detector_backend="opencv", model_name="Facenet512"):
    try:
        reps = DeepFace.represent(
            img_path=img_bgr,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True
        )
        if isinstance(reps, list) and reps:
            emb = np.array(reps[0]["embedding"], dtype=np.float32)
            return _normalize(emb)
    except Exception:
        return None
    return None

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:

    a = _normalize(a)
    b = _normalize(b)
    return 1.0 - float(np.dot(a, b))

def enroll_average(embeddings: list[np.ndarray]) -> np.ndarray | None:
    if not embeddings:
        return None
    E = np.stack(embeddings, axis=0)
    mean = E.mean(axis=0)
    return _normalize(mean)

def verify(probe: np.ndarray, reference: np.ndarray, thr: float = 0.40) -> tuple[bool, float]:
    if probe is None or reference is None:
        return False, 1.0
    dist = cosine_distance(probe, reference)
    return (dist <= thr), float(dist)
