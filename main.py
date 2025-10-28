# streamlit_app.py — Streamlit 용 One-Click Poster
import os, math, random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import hsv_to_rgb

# ----- Settings -----
st.set_page_config(page_title="Generative Poster", layout="wide")  # 넓은 레이아웃
st.title("🎨 Generative Poster (Streamlit)")

with st.sidebar:
    st.header("Controls")
    n_layers = st.slider("Layers", 3, 20, 8, 1)
    wobble = st.slider("Wobble", 0.01, 9.0, 0.18, 0.01)
    palette_mode = st.selectbox("Palette", ["pastel", "vivid", "mono", "random"])
    base_h = st.slider("Mono Hue (0~1)", 0.0, 1.0, 0.60, 0.01)
    save_poster = st.checkbox("Save poster", value=True)
    save_dir = st.text_input("Save dir", "posters")
    preview_width = st.slider("Preview width (px)", 600, 1900, 1200, 50)  # ★ 처음부터 크게

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
    seed = int(datetime.now().strftime("%H%M%S%f")) % 10_000_000
    random.seed(seed); np.random.seed(seed)

    # ★ 포스터 비율을 크게 (인치) — 출력은 st.image(width=...)로 제어
    fig, ax = plt.subplots(figsize=(10, 14))
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
            transform=ax.transAxes, fontsize=16, weight="bold")

    saved_path = None
    if save:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"poster_{palette_mode}_{seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
        saved_path = os.path.join(save_dir, fname)
        plt.savefig(saved_path, dpi=300, bbox_inches="tight", pad_inches=0.0)

    # 파일 저장 안 하더라도, 즉시 미리보기 위해 임시 버퍼 저장
    if not saved_path:
        os.makedirs("tmp", exist_ok=True)
        saved_path = os.path.join("tmp", f"preview_{palette_mode}_{seed}.png")
        plt.savefig(saved_path, dpi=200, bbox_inches="tight", pad_inches=0.0)

    return saved_path

# --- 최초 진입 시 자동 생성 + 큰 크기로 표시 ---
if "did_first_render" not in st.session_state:
    st.session_state.did_first_render = True
    auto_run = True
else:
    auto_run = False

col_btn = st.container()  # 버튼은 위에, 이미지는 아래 전체 폭
with col_btn:
    regenerate = st.button("🔁 Regenerate", use_container_width=True)

# 이미지 들어갈 자리
img_slot = st.empty()

# 실행 조건: 첫 로드 자동 생성 OR 사용자가 재생성 클릭
if auto_run or regenerate:
    path = generate_poster(n_layers, wobble, palette_mode, base_h, save_poster, save_dir)
    # ★ pyplot 대신 image로 ‘큰 폭’ 렌더 — 클릭 확대 없이 처음부터 크게
    img_slot.image(path, width=preview_width)     # ← 여기서 크기 고정
    st.success(f"Saved: {path}" if save_poster else f"Preview: {path}")
