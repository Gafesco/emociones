import tkinter as tk
from tkinter import ttk
import webbrowser
import urllib.parse

PALETTE = {
    "bg":         "#0E1A2B",   # fondo app
    "card":       "#12233A",   # tarjeta/sección
    "shadow":     "#0A1422",   # sombra sutil
    "accent":     "#2E6AF2",   # azul principal (header)
    "accent_2":   "#1F56D6",
    "text":       "#E9EEF7",   # texto principal
    "muted":      "#AFC1DA",
    "border":     "#1C3254",
    "btn_pri":    "#2E6AF2",   # botón primario
    "btn_pri_h":  "#3A77FF",
    "btn_sec":    "#183055",   # botón secundario
    "btn_sec_h":  "#214171",
}

YOUTUBE_SEARCH = "https://www.youtube.com/results?search_query="
GOOGLE_SEARCH  = "https://www.google.com/search?q="

def _yt_url(q: str) -> str:
    return YOUTUBE_SEARCH + urllib.parse.quote(q)

def _gg_url(q: str) -> str:
    return GOOGLE_SEARCH + urllib.parse.quote(q)

def _label_of(item):
    if isinstance(item, dict):
        return item.get("label") or item.get("url") or ""
    return str(item)

def _url_music(item):
    if isinstance(item, dict) and item.get("url"):
        return item["url"]
    return _yt_url(_label_of(item))

def _url_movie(item):
    if isinstance(item, dict) and item.get("url"):
        return item["url"]
    return _yt_url(f"{_label_of(item)} trailer")

def _url_activity(item):
    if isinstance(item, dict) and item.get("url"):
        return item["url"]
    return _gg_url(_label_of(item))


class RoundedButton(tk.Canvas):
    def __init__(self, master, text, command=None,
                 fill=PALETTE["btn_pri"], hover_fill=PALETTE["btn_pri_h"],
                 fg="#FFFFFF", container_bg=PALETTE["card"],
                 radius=12, pad_x=22, pad_y=12, font=("Segoe UI", 12, "bold"),
                 shadow=True, **kwargs):
        super().__init__(master, highlightthickness=0, bd=0, bg=container_bg, **kwargs)
        self.command = command
        self.fill = fill
        self.hover_fill = hover_fill
        self.fg = fg
        self.radius = radius
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.font = font
        self.shadow = shadow
        self.text_value = text

        tmp = tk.Label(self, text=text, font=font)
        tmp.update_idletasks()
        w = tmp.winfo_reqwidth() + pad_x * 2
        h = tmp.winfo_reqheight() + pad_y * 2
        tmp.destroy()

        self.configure(width=w + 8, height=h + 8)
        self._draw(self.fill)
        self.bind("<Enter>", lambda e: self._draw(self.hover_fill))
        self.bind("<Leave>", lambda e: self._draw(self.fill))
        self.bind("<Button-1>", self._on_click)

    def _round_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1+r, y1, x2-r, y1, x2, y1, x2, y1+r,
            x2, y2-r, x2, y2, x2-r, y2, x1+r, y2,
            x1, y2, x1, y2-r, x1, y1+r, x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _draw(self, fill_color):
        self.delete("all")
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        if self.shadow:
            self._round_rect(5, 5, w-1, h-1, self.radius, fill=PALETTE["shadow"], outline="")
        self._round_rect(2, 2, w-4, h-4, self.radius, fill=fill_color, outline="")
        self.create_text(w//2, h//2, text=self.text_value, fill=self.fg, font=self.font)

    def _on_click(self, _):
        if callable(self.command):
            self.command()


def _init_styles():
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure("App.TFrame",    background=PALETTE["bg"])
    style.configure("Header.TFrame", background=PALETTE["accent"])
    style.configure("Body.TFrame",   background=PALETTE["bg"])
    style.configure("Card.TFrame",   background=PALETTE["card"], borderwidth=1, relief="flat")
    style.configure("CardPad.TFrame",background=PALETTE["bg"])

    style.configure("Header.TLabel", background=PALETTE["accent"], foreground="#FFFFFF",
                    font=("Segoe UI", 22, "bold"))

    style.configure("H2.TLabel",     background=PALETTE["card"], foreground=PALETTE["text"],
                    font=("Segoe UI Semibold", 16))
    style.configure("Body.TLabel",   background=PALETTE["card"], foreground=PALETTE["text"],
                    font=("Segoe UI", 13), wraplength=540, justify="left")
    style.configure("Hint.TLabel",   background=PALETTE["bg"], foreground=PALETTE["muted"],
                    font=("Segoe UI", 11))

    style.configure("Blue.Vertical.TScrollbar",
                    troughcolor=PALETTE["card"],
                    background=PALETTE["accent_2"],
                    arrowcolor="#FFFFFF", bordercolor=PALETTE["bg"])
    return style


def open_reco_window(emotion: str, recos: dict):
    music_list  = list(recos.get("music", []))
    movies_list = list(recos.get("movies", []))
    acts_list   = list(recos.get("activities", []))
    idx = {"music": 0, "movies": 0, "activities": 0}

    root = tk.Tk()
    root.title(f"Recomendaciones — {emotion.upper()}")
    root.geometry("820x620")
    root.minsize(760, 560)
    root.configure(bg=PALETTE["bg"])

    _init_styles()

    # ======= Header  =======
    header = ttk.Frame(root, style="Header.TFrame")
    header.pack(fill="x")
    header.grid_columnconfigure(0, weight=1)
    title = ttk.Label(header, text=f"Emoción detectada: {emotion.upper()}",
                      style="Header.TLabel", anchor="center")
    title.grid(row=0, column=0, pady=10, sticky="ew")

    # ======= Body con scroll =======
    body = ttk.Frame(root, style="Body.TFrame")
    body.pack(fill="both", expand=True)

    canvas = tk.Canvas(body, highlightthickness=0, bg=PALETTE["bg"])
    scrollbar = ttk.Scrollbar(body, orient="vertical",
                              command=canvas.yview, style="Blue.Vertical.TScrollbar")
    scroll_frame = ttk.Frame(canvas, style="Body.TFrame")

    def on_configure(_):
        canvas.configure(scrollregion=canvas.bbox("all"))
    scroll_frame.bind("<Configure>", on_configure)

    window_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

    def on_canvas_resize(event):
        canvas.itemconfigure(window_id, width=event.width)
    canvas.bind("<Configure>", on_canvas_resize)

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True, padx=(16, 0), pady=16)
    scrollbar.pack(side="right", fill="y", padx=(0, 16), pady=16)

    # ======= Helpers =======
    def _set_text(widget: ttk.Label, text: str):
        widget.config(text=("• " + text if text else "—"))

    def _current(lst, k):
        if not lst:
            return None
        return lst[idx[k] % len(lst)]

    def _next(lst, k, label_widget):
        if not lst:
            return
        idx[k] = (idx[k] + 1) % len(lst)
        item = _current(lst, k)
        _set_text(label_widget, _label_of(item))

    def _card(parent):
        pad = ttk.Frame(parent, style="CardPad.TFrame")
        pad.pack(fill="x", pady=10)
        shadow = tk.Canvas(pad, height=6, highlightthickness=0, bg=PALETTE["bg"])
        shadow.pack(fill="x")
        card = ttk.Frame(pad, style="Card.TFrame")
        card.pack(fill="x")
        return card

    def _adjust_wrap(label: ttk.Label, row: ttk.Frame, btns: ttk.Frame, min_wrap=260, pad=36):

        row.update_idletasks()
        btns.update_idletasks()
        row_w = row.winfo_width()
        btn_w = btns.winfo_width() or btns.winfo_reqwidth()
        wrap = max(min_wrap, row_w - btn_w - pad)
        label.configure(wraplength=wrap)

    def build_card(title_text, items, key, open_url_fn, primary_text):
        card = _card(scroll_frame)
        card.grid_columnconfigure(0, weight=1)

        ttk.Label(card, text=title_text, style="H2.TLabel").grid(row=0, column=0, sticky="w", padx=20, pady=(16, 6))

        row = ttk.Frame(card, style="Card.TFrame")
        row.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        row.grid_columnconfigure(0, weight=1)

        label = ttk.Label(row, text="", style="Body.TLabel", justify="left")
        label.grid(row=0, column=0, sticky="w")

        btns = ttk.Frame(row, style="Card.TFrame")
        btns.grid(row=0, column=1, sticky="e", padx=(16, 0))

        RoundedButton(btns, text="⟳  Otro",
                      command=lambda: _next(items, key, label),
                      fill=PALETTE["btn_sec"], hover_fill=PALETTE["btn_sec_h"],
                      container_bg=PALETTE["card"],
                      font=("Segoe UI", 12, "bold")).pack(side="left", padx=(0, 12))

        RoundedButton(btns, text=f"▶  {primary_text}",
                      command=lambda: (webbrowser.open(open_url_fn(_current(items, key))) if items else None),
                      fill=PALETTE["btn_pri"], hover_fill=PALETTE["btn_pri_h"],
                      container_bg=PALETTE["card"],
                      font=("Segoe UI", 12, "bold")).pack(side="left")

        # inicializar texto
        _set_text(label, _label_of(_current(items, key)) if items else "")

        row.bind("<Configure>", lambda e: _adjust_wrap(label, row, btns))

    # ======= Tarjetas =======
    build_card("Música",      music_list,  "music",      _url_music,    "Reproducir")
    build_card("Películas",   movies_list, "movies",     _url_movie,    "Tráiler")
    build_card("Actividades", acts_list,   "activities", _url_activity, "Ver guía")

    # ======= Footer =======
    footer = ttk.Frame(root, style="App.TFrame")
    footer.pack(fill="x", pady=(0, 14), padx=16)
    ttk.Label(footer, text="Usa “⟳  Otro” para cambiar de recomendación.",
              style="Hint.TLabel").pack(side="left")

    RoundedButton(footer, text="Cerrar",
                  command=root.destroy,
                  fill=PALETTE["btn_pri"], hover_fill=PALETTE["btn_pri_h"],
                  container_bg=PALETTE["bg"],
                  font=("Segoe UI", 12, "bold"), pad_x=28, pad_y=10).pack(side="right")

    root.mainloop()
