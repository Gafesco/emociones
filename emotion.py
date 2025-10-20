import os, time, cv2, numpy as np
from deepface import DeepFace

EMO_MAP = {
    "angry": "enojado",
    "disgust": "enojado",
    "fear": "asustado",
    "happy": "feliz",
    "sad": "triste",
    "surprise": "sorprendido",
    "neutral": "neutral",
}

class FaceDetector:
    def __init__(self, scale_factor=1.2, min_neighbors=5, min_size=(80, 80)):
        haar_dir = cv2.data.haarcascades
        xml_path = os.path.join(haar_dir, "haarcascade_frontalface_default.xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"No se encontró el Haarcascade en {xml_path}")
        self.detector = cv2.CascadeClassifier(xml_path)
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size = tuple(min_size)

    def detect(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors, minSize=self.min_size
        )
        return [(int(x),int(y),int(x+w),int(y+h)) for (x,y,w,h) in faces]

def _crop_with_padding(img, box, pad_ratio=0.25):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pw, ph = int(bw*pad_ratio), int(bh*pad_ratio)
    x1, y1 = max(0,x1-pw), max(0,y1-ph)
    x2, y2 = min(w,x2+pw), min(h,y2+ph)
    return img[y1:y2, x1:x2]

class EmotionClassifierDeepFace:
    def __init__(self, interval_sec: float = 0.6, smoothing: int = 5):
        self.interval_sec = float(interval_sec)
        self.smoothing = int(smoothing)
        self.last_ts = 0.0
        self.history = []
        self.last_output = "neutral"
        self.last_probs = {}

    def _now(self): return time.time()

    def _smoothed(self):
        if not self.history: return "neutral"
        vals, counts = np.unique(self.history, return_counts=True)
        return vals[int(np.argmax(counts))]

    def predict(self, frame_bgr, face_boxes=None):
        t = self._now()
        if t - self.last_ts < self.interval_sec:
            return self.last_output
        self.last_ts = t

        if not face_boxes:
            self.history.append("neutral")
            if len(self.history) > self.smoothing: self.history.pop(0)
            self.last_output = self._smoothed()
            self.last_probs = {}
            return self.last_output

        areas = [ (x2-x1)*(y2-y1) for (x1,y1,x2,y2) in face_boxes ]
        box = face_boxes[int(np.argmax(areas))]
        roi = _crop_with_padding(frame_bgr, box, pad_ratio=0.25)

        try:
            result = DeepFace.analyze(
                img_path=roi, actions=["emotion"],
                detector_backend="opencv", enforce_detection=False
            )
            if isinstance(result, list): result = result[0]
            dom = result.get("dominant_emotion","neutral")
            probs = result.get("emotion",{})
        except Exception:
            dom, probs = "neutral", {}

        mapped = EMO_MAP.get(dom, "neutral")
        self.history.append(mapped)
        if len(self.history) > self.smoothing: self.history.pop(0)

        self.last_output = self._smoothed()
        self.last_probs = {EMO_MAP.get(k,k): float(v) for k,v in probs.items()}
        return self.last_output
