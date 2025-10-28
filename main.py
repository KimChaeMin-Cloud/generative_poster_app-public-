# streamlit_app.py — Streamlit 용 One-Click Poster
import os, math, random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import hsv_to_rgb

# ----- Settings (좌측 사이드바로 제어) -----
# ★ wide 레이아웃로 전환
st.set_page_config(page_title="Generative Poster", layout="wide")
st.title("🎨 Generative Poster (Streamlit)")

with st.sidebar:
    st.header("Controls")
    n_layers = st.slider("Layers", 3, 20, 8, 1)
    wobble = st.slider("Wobble", 0.01, 9.0, 0.18, 0.01)
    palette_mode = st.selectbox("Palette", ["pastel", "vivid", "mono", "random"])
    base_h = st.slider("Mono Hue (0~1)", 0.0, 1.0, 0.60, 0.01)
    save_poster = st.checkbox("Save poster", value=True)
    save_dir = st.text_input("Save dir", "posters")
    # ★ 미리보기 확대 배율 (그림 실제 해상도와는 무관, 화면 채움용)
    preview_scale = st.slider("Preview scale", 1.0, 2.0, 1.4, 0.1)

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

def generate_poster(n_layers, wobble, palette_mode, base_h, save, save_dir, preview_scale=1.4):
    # seed를 시간기반으로 고쳐 매 클릭마다 새 결과
    seed = int(datetime.now().strftime("%H%M%S%f")) % 10_000_000
    random.seed(seed); np.random.seed(seed)

    # ★ 기본 캔버스를 크게 생성 (세로형 포스터 비율)
    base_figsize = (10, 14)  # (가로, 세로) 인치
    figsize = (base_figsize[0]*preview_scale, base_figsize[1]*preview_scale)

    fig, ax = plt.subplots(figsize=figsize)
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
            transform=ax.transAxes, fontsize=16, weight="bold")  # ★ 자막도 조금 키움

    saved_path = None
    if save:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"poster_{palette_mode}_{seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
        saved_path = os.path.join(save_dir, fname)
        # ★ 저장은 고해상도로
        plt.savefig(saved_path, dpi=300, bbox_inches="tight", pad_inches=0.0)

    return fig, saved_path

# --- One-Click 버튼 ---
# (버튼은 좌측 폭 제한의 영향을 받을 수 있어 단일 컬럼으로 배치)
clicked = st.button("🎨 Generate Poster", use_container_width=True)

placeholder = st.empty()

if clicked:
    fig, saved = generate_poster(n_layers, wobble, palette_mode, base_h, save_poster, save_dir, preview_scale)
    with placeholder.container():
        # ★ 컨테이너 폭에 맞춰 크게 표시
        st.pyplot(fig, clear_figure=True, use_container_width=True)
        if saved:
            st.success(f"Saved: {saved}")
