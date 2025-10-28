# streamlit_app.py — Streamlit 용 One-Click Poster
import os, math, random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import hsv_to_rgb

# ----- Settings (좌측 사이드바로 제어) -----
st.set_page_config(page_title="Generative Poster", layout="centered")
st.title("🎨 Generative Poster (Streamlit)")

with st.sidebar:
    st.header("Controls")
    n_layers = st.slider("Layers", 3, 20, 8, 1)
    wobble = st.slider("Wobble", 0.01, 9.0, 0.18, 0.01)
    palette_mode = st.selectbox("Palette", ["pastel", "vivid", "mono", "random"])
    base_h = st.slider("Mono Hue (0~1)", 0.0, 1.0, 0.60, 0.01)
    save_poster = st.checkbox("Save poster", value=True)
    save_dir = st.text_input("Save dir", "posters")

def blob(center=(0.5, 0.5), r=0.3, points=200, wobble=0.15):
    angles = np.linspace(0, 2*math.pi, points, endpoint=False)
    radii  = r * (1 + wobble*(np.random.rand(points)-0.5))
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return x, y

def make_palette(k=6, mode="pastel", base_h=0.60):
    cols = []
    for _ in range(k):
        if mode == "pastel":
            h = random.random(); s = random.uniform(0.15,0.35); v = random.uniform(0.9,1.0)
        elif mode == "vivid":
            h = random.random(); s = random.uniform(0.8,1.0);  v = random.uniform(0.8,1.0)
        elif mode == "mono":
            h = base_h;         s = random.uniform(0.2,0.6);   v = random.uniform(0.5,1.0)
        else:  # random
            h = random.random(); s = random.uniform(0.3,1.0); v = random.uniform(0.5,1.0)
        cols.append(tuple(hsv_to_rgb([h,s,v])))
    return cols

def generate_poster(n_layers, wobble, palette_mode, base_h, save, save_dir):
    # seed를 시간기반으로 고쳐 매 클릭마다 새 결과
    seed = int(datetime.now().strftime("%H%M%S%f")) % 10_000_000
    random.seed(seed); np.random.seed(seed)

    fig, ax = plt.subplots(figsize=(6,8))
    ax.axis("off")
    ax.set_facecolor((0.97,0.97,0.97))

    palette = make_palette(6, mode=palette_mode, base_h=base_h)
    for _ in range(n_layers):
        cx, cy = random.random(), random.random()
        rr = random.uniform(0.15, 0.45)
        x, y = blob((cx,cy), r=rr, wobble=wobble)
        color = random.choice(palette)
        alpha = random.uniform(0.3, 0.6)
        ax.fill(x, y, color=color, alpha=alpha, edgecolor=(0,0,0,0))

    ax.text(0.05, 0.95, f"Poster • {palette_mode}",
            transform=ax.transAxes, fontsize=12, weight="bold")

    saved_path = None
    if save:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"poster_{palette_mode}_{seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
        saved_path = os.path.join(save_dir, fname)
        plt.savefig(saved_path, dpi=150, bbox_inches="tight")

    return fig, saved_path

# --- One-Click 버튼 ---
col1, col2 = st.columns([1, 2])
with col1:
    clicked = st.button("🎨 Generate Poster", use_container_width=True)

placeholder = st.empty()

if clicked:
    fig, saved = generate_poster(n_layers, wobble, palette_mode, base_h, save_poster, save_dir)
    with placeholder.container():
        st.pyplot(fig, clear_figure=True)
        if saved:
            st.success(f"Saved: {saved}")
