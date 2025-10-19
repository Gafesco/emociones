# app.py
import os
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from camera import VideoStream
from emotion import FaceDetector, EmotionClassifierDeepFace
from reco_engine import RecoEngine
from ui import open_reco_window

# ------------------- CONFIG -------------------
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
BG_PATH    = os.path.join(ASSETS_DIR, "ui_bg.jpg")   # tu fondo
ICONS_DIR  = os.path.join(ASSETS_DIR, "icons")

# Mapea emociones -> icono (jpg/png). Si no existe, caerá a texto.
ICON_MAP = {
    "feliz":       "feliz.jpg",
    "triste":      "triste.jpg",
    "enojado":     "enojado.jpg",
    "sorprendido": "sorprendido.jpg",
    "asustado":    "asustado.jpg",
    "neutral":     "neutral.jpg",
}

# Coordenadas (x, y, w, h) **sobre la imagen de fondo**
CAMERA_REGION = (50, 210, 780, 500)   # rectángulo grande blanco (cámara)
EMO_REGION    = (917, 335, 280, 365)  # rectángulo pequeño blanco (icono de emoción)

# ------------------- DIBUJO DE TEXTO -------------------
def _find_font():
    candidates = [
        r"C:\Windows\Fonts\SegoeUI.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def draw_text_utf8(img_bgr, text, pos, font_size=30, color=(20, 20, 20), anchor="lt"):
    """Dibuja texto UTF-8 con Pillow. anchor: 'lt' (left-top), 'mm' (center)."""
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    fp = _find_font()
    font = ImageFont.truetype(fp, font_size) if fp else ImageFont.load_default()
    color_rgb = (int(color[2]), int(color[1]), int(color[0]))
    if anchor == "mm":
        text_w = draw.textlength(text, font=font)
        text_h = font.size
        pos = (pos[0] - text_w / 2, pos[1] - text_h / 2)
    draw.text(pos, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ------------------- UTILIDADES DE LAYOUT -------------------
def fit_into(src_bgr, dst_wh):
    """Reescala src_bgr para caber dentro de dst_wh (w,h) manteniendo aspecto."""
    dst_w, dst_h = dst_wh
    sh, sw = src_bgr.shape[:2]
    scale = min(dst_w / sw, dst_h / sh) if sw > 0 and sh > 0 else 1.0
    new_w, new_h = max(1, int(sw * scale)), max(1, int(sh * scale))
    resized = cv2.resize(src_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    x0 = (dst_w - new_w) // 2
    y0 = (dst_h - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

def fit_into_info(src_bgr, dst_wh):
    """
    Igual que fit_into pero también devuelve (x0, y0, scale, new_w, new_h)
    para poder mapear coordenadas (ej. rectángulos de rostro).
    """
    dst_w, dst_h = dst_wh
    sh, sw = src_bgr.shape[:2]
    scale = min(dst_w / sw, dst_h / sh) if sw > 0 and sh > 0 else 1.0
    new_w, new_h = max(1, int(sw * scale)), max(1, int(sh * scale))
    resized = cv2.resize(src_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    x0 = (dst_w - new_w) // 2
    y0 = (dst_h - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas, x0, y0, scale, new_w, new_h

def paste_region(bg_bgr, content_bgr, region_xywh):
    """
    Pega content_bgr dentro de bg_bgr en la región (x,y,w,h), respetando límites.
    Reescala content_bgr para encajar y recorta si es necesario.
    """
    x, y, w, h = region_xywh
    H, W = bg_bgr.shape[:2]
    x2, y2 = min(x + w, W), min(y + h, H)
    w_real, h_real = x2 - x, y2 - y
    if w_real <= 0 or h_real <= 0:
        return bg_bgr
    content_fit = fit_into(content_bgr, (w_real, h_real))
    bg_bgr[y:y+h_real, x:x+w_real] = content_fit
    return bg_bgr

def load_icon_for(emotion):
    fn = ICON_MAP.get(emotion)
    if not fn:
        return None
    path = os.path.join(ICONS_DIR, fn)
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # por si trae alpha
    if img is None:
        return None
    # si tiene canal alpha (PNG), componer sobre blanco
    if img.shape[2] == 4:
        alpha = img[:, :, 3] / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        bg = np.full_like(rgb, 255, dtype=np.float32)
        out = (rgb * alpha[..., None] + bg * (1 - alpha[..., None])).astype(np.uint8)
        return out
    return img

# ------------------- MAIN -------------------
if __name__ == "__main__":
    # Cargar fondo
    bg0 = cv2.imread(BG_PATH)
    if bg0 is None:
        raise FileNotFoundError(f"No se pudo cargar el fondo: {BG_PATH}")

    # Inicializar módulos
    vs = VideoStream(width=1280, height=720)
    fd = FaceDetector()
    ec = EmotionClassifierDeepFace(interval_sec=0.6, smoothing=5)
    reco = RecoEngine()

    # Ventana
    window_name = "Sistema Inteligente de Recomendaciones Emocionales"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, bg0.shape[1], bg0.shape[0])

    # Control de ventana de recomendaciones
    STABLE_SEC = 5.0
    COOLDOWN_SEC = 3.0
    last_emotion = None
    recos_cache = {"music": [], "movies": [], "activities": []}
    last_change_ts = time.time()
    last_window_ts = 0.0

    # Cache de iconos
    icon_cache = {}

    try:
        while True:
            ok, frame = vs.read()
            if not ok:
                break

            # Clonar fondo (canvas base)
            canvas = bg0.copy()

            # Detectar rostros
            boxes = fd.detect(frame)

            # --- Elegir SOLO UN rostro (el más grande) para evitar conflictos ---
            main_box = None
            if boxes:
                areas = [(x2-x1)*(y2-y1) for (x1,y1,x2,y2) in boxes]
                main_box = boxes[int(np.argmax(areas))]
                boxes_for_emotion = [main_box]  # solo 1
            else:
                boxes_for_emotion = []

            # Predicción de emoción usando SOLO el rostro principal
            emo = ec.predict(frame, face_boxes=boxes_for_emotion)

            # Si cambia la emoción → refresca recomendaciones (para la ventana secundaria)
            if emo != last_emotion:
                recos_cache = reco.get(emo, k=None)  # todas, barajadas
                last_emotion = emo
                last_change_ts = time.time()

            # 1) Preparar frame para región de cámara y dibujar el recuadro ahí (mapeando coords)
            x_cam, y_cam, w_cam, h_cam = CAMERA_REGION
            frame_fit, off_x, off_y, scale, new_w, new_h = fit_into_info(frame, (w_cam, h_cam))

            # Dibujar el rectángulo SOLO si hay rostro principal
            if main_box is not None:
                x1, y1, x2, y2 = main_box
                # escalar y desplazar al espacio de frame_fit (w_cam x h_cam)
                rx1 = int(off_x + x1 * scale)
                ry1 = int(off_y + y1 * scale)
                rx2 = int(off_x + x2 * scale)
                ry2 = int(off_y + y2 * scale)
                # clamp a límites del frame_fit
                rx1, ry1 = max(0, rx1), max(0, ry1)
                rx2, ry2 = min(w_cam-1, rx2), min(h_cam-1, ry2)
                cv2.rectangle(frame_fit, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

            # Pegar el frame (con recuadro dibujado) en el canvas
            # (como ya está en el tamaño exacto de la región, no se deforma)
            x2, y2 = min(x_cam + w_cam, canvas.shape[1]), min(y_cam + h_cam, canvas.shape[0])
            w_real, h_real = x2 - x_cam, y2 - y_cam
            if w_real > 0 and h_real > 0:
                canvas[y_cam:y_cam+h_real, x_cam:x_cam+w_real] = frame_fit[:h_real, :w_real]

            # 2) Icono/texto de emoción en su región (robusto a límites)
            if emo not in icon_cache:
                icon_cache[emo] = load_icon_for(emo)
            icon = icon_cache.get(emo)

            if icon is not None:
                canvas = paste_region(canvas, icon, EMO_REGION)
            else:
                # Fallback: escribir texto centrado dentro del área disponible
                x, y, w, h = EMO_REGION
                H, W = canvas.shape[:2]
                x2, y2 = min(x + w, W), min(y + h, H)
                w_real, h_real = max(0, x2 - x), max(0, y2 - y)
                cx, cy = x + w_real // 2, y + h_real // 2
                canvas = draw_text_utf8(canvas, emo.upper(), (cx, cy),
                                        font_size=48, color=(30, 30, 30), anchor="mm")

            # Mostrar ventana
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF

            # Abrir recomendaciones manualmente
            if key in (ord('r'), ord('R')):
                open_reco_window(emo, recos_cache)
                last_window_ts = time.time()
                continue

            # Apertura automática si emoción estable y hay rostro
            now = time.time()
            if (now - last_change_ts) >= STABLE_SEC and (now - last_window_ts) >= COOLDOWN_SEC and main_box is not None:
                open_reco_window(emo, recos_cache)
                last_window_ts = time.time()

            if key == 27:  # ESC
                break

    finally:
        vs.release()
        cv2.destroyAllWindows()
