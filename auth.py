# auth.py
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np

from camera import VideoStream
from emotion import FaceDetector
from face_id import face_embedding, enroll_average, verify
from accounts import get_user_embedding, set_user_embedding, user_exists

PALETTE = {
    "bg": "#0E1A2B",
    "card": "#12233A",
    "accent": "#2E6AF2",
    "text": "#E9EEF7",
    "muted": "#AFC1DA",
    "ok": "#28C76F",
    "warn": "#FF9F43",
    "err": "#EA5455",
}

SAMPLES_REQUIRED = 5
THRESHOLD = 0.40  # ajústalo tras pruebas

class AuthWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Acceso — Reconocimiento Facial")
        self.geometry("900x560")
        self.configure(bg=PALETTE["bg"])
        self.resizable(False, False)

        # cámara y detector
        self.vs = VideoStream(width=960, height=540)
        self.fd = FaceDetector()

        # UI
        self._build_ui()

        # loop de cámara
        self._frame = None
        self._update_video()

        # estado registro
        self.samples = []
        self.sample_count = 0

        self.result_username = None  # se setea al hacer login OK

    def destroy(self):
        try:
            self.vs.release()
        except Exception:
            pass
        super().destroy()

    # ---------- UI ----------
    def _build_ui(self):
        style = ttk.Style()
        try: style.theme_use("clam")
        except Exception: pass

        style.configure("TFrame", background=PALETTE["bg"])
        style.configure("Card.TFrame", background=PALETTE["card"])
        style.configure("TLabel", background=PALETTE["card"], foreground=PALETTE["text"], font=("Segoe UI", 11))
        style.configure("Title.TLabel", background=PALETTE["bg"], foreground=PALETTE["text"], font=("Segoe UI", 16, "bold"))
        style.configure("Heading.TLabel", background=PALETTE["card"], foreground=PALETTE["text"], font=("Segoe UI", 13, "bold"))
        style.configure("TEntry", fieldbackground="#0E1A2B", foreground="#FFFFFF")
        style.configure("TButton", font=("Segoe UI", 10, "bold"))

        left = ttk.Frame(self, style="TFrame")
        right = ttk.Frame(self, style="TFrame")
        left.pack(side="left", fill="both", expand=True, padx=(16,8), pady=16)
        right.pack(side="right", fill="y", padx=(8,16), pady=16)

        # Video card
        video_card = ttk.Frame(left, style="Card.TFrame")
        video_card.pack(fill="both", expand=True)
        ttk.Label(video_card, text="Coloca tu rostro frente a la cámara", style="Heading.TLabel").pack(anchor="w", padx=12, pady=(12,6))
        self.video_label = ttk.Label(video_card)
        self.video_label.pack(fill="both", expand=True, padx=12, pady=(0,12))

        # Tabs (Login / Registrar)
        ttk.Label(right, text="Acceso al sistema", style="Title.TLabel").pack(anchor="w", pady=(0,10))
        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)

        # --- Login tab
        login_tab = ttk.Frame(nb, style="Card.TFrame")
        nb.add(login_tab, text="Iniciar sesión")

        ttk.Label(login_tab, text="Usuario", style="TLabel").pack(anchor="w", padx=12, pady=(12,4))
        self.login_user = ttk.Entry(login_tab, width=26)
        self.login_user.pack(padx=12, pady=(0,8))

        self.login_status = ttk.Label(login_tab, text="", style="TLabel")
        self.login_status.pack(anchor="w", padx=12, pady=(4,8))

        ttk.Button(login_tab, text="Iniciar con rostro", command=self._do_login).pack(padx=12, pady=10)

        # --- Register tab
        reg_tab = ttk.Frame(nb, style="Card.TFrame")
        nb.add(reg_tab, text="Crear cuenta")

        ttk.Label(reg_tab, text="Nuevo usuario", style="TLabel").pack(anchor="w", padx=12, pady=(12,4))
        self.reg_user = ttk.Entry(reg_tab, width=26)
        self.reg_user.pack(padx=12, pady=(0,8))

        self.reg_status = ttk.Label(reg_tab, text="Capturas: 0 / " + str(SAMPLES_REQUIRED), style="TLabel")
        self.reg_status.pack(anchor="w", padx=12, pady=(4,6))

        ttk.Button(reg_tab, text="Capturar muestra", command=self._capture_sample).pack(padx=12, pady=(4,6))
        ttk.Button(reg_tab, text="Guardar registro", command=self._save_registration).pack(padx=12, pady=(4,10))

        ttk.Label(reg_tab, text="Mira al Frente",
                  style="TLabel").pack(anchor="w", padx=12, pady=(0,12))

        # Footer
        ttk.Label(self, text="Requiere usuario + rostro. Si falla, verifica luz, encuadre y vuelve a intentar.",
                  style="Title.TLabel").pack(side="bottom", pady=(0,8))

    # ---------- Cámara ----------
    def _update_video(self):
        ok, frame = self.vs.read()
        if ok:
            # dibujar rostro (el más grande)
            boxes = self.fd.detect(frame)
            if boxes:
                areas = [ (x2-x1)*(y2-y1) for (x1,y1,x2,y2) in boxes ]
                x1,y1,x2,y2 = boxes[int(np.argmax(areas))]
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            # convertir a ImageTk
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self._frame = frame
        self.after(20, self._update_video)

    # ---------- Lógica ----------
    def _get_current_embedding(self):
        if self._frame is None:
            return None
        return face_embedding(self._frame)  # usa detector interno de DeepFace (opencv)

    def _do_login(self):
        username = self.login_user.get().strip()
        if not username:
            messagebox.showerror("Error", "Ingresa tu usuario.")
            return
        ref = get_user_embedding(username)
        if ref is None:
            messagebox.showerror("Error", "Usuario no encontrado.")
            return
        probe = self._get_current_embedding()
        if probe is None:
            self.login_status.configure(text="No se detectó rostro. Ajusta la cámara.", foreground=PALETTE["warn"])
            return
        ok, dist = verify(probe, ref, thr=THRESHOLD)
        if ok:
            self.login_status.configure(text="Acceso correcto", foreground=PALETTE["ok"])
            self.result_username = username
            self.after(700, self.destroy)  # le da un poco más de tiempo (0.7 seg)
        else:
            self.login_status.configure(text=f"Rostro no coincide (dist={dist:.3f}). Intenta de nuevo.",
                                        foreground=PALETTE["err"])

    def _capture_sample(self):
        username = self.reg_user.get().strip()
        if not username:
            messagebox.showerror("Error", "Ingresa un usuario para registrar.")
            return
        if user_exists(username):
            messagebox.showerror("Error", "Ese usuario ya existe. Usa otro nombre.")
            return
        emb = self._get_current_embedding()
        if emb is None:
            self.reg_status.configure(text="No se detectó rostro. Intenta acercarte o mejora la luz.", foreground=PALETTE["warn"])
            return
        self.samples.append(emb)
        self.sample_count += 1
        self.reg_status.configure(text=f"Capturas: {self.sample_count} / {SAMPLES_REQUIRED}", foreground=PALETTE["text"])

    def _save_registration(self):
        username = self.reg_user.get().strip()
        if not username:
            messagebox.showerror("Error", "Ingresa un usuario para registrar.")
            return
        if self.sample_count < SAMPLES_REQUIRED:
            messagebox.showerror("Incompleto", f"Necesitas al menos {SAMPLES_REQUIRED} capturas. Lleva {self.sample_count}.")
            return
        mean_emb = enroll_average(self.samples)
        if mean_emb is None:
            messagebox.showerror("Error", "No fue posible procesar las muestras.")
            return
        set_user_embedding(username, mean_emb)
        messagebox.showinfo("Listo", f"Usuario '{username}' registrado correctamente.")
        # reset
        self.samples.clear()
        self.sample_count = 0
        self.reg_status.configure(text=f"Capturas: 0 / {SAMPLES_REQUIRED}", foreground=PALETTE["text"])

def run_auth() -> str | None:
    """
    Abre la ventana de autenticación.
    Devuelve el nombre de usuario si el login fue exitoso; None en otro caso.
    """
    app = AuthWindow()
    app.mainloop()
    return app.result_username
