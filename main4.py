# streamlit_app.py — One-Click Poster
# (+ Seed 고정, + 배치 생성 ZIP, + PNG/SVG 저장 & DPI 조절)
import os, io, math, random, zipfile
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import hsv_to_rgb

# ----- Settings -----
st.set_page_config(page_title="Generative Poster", layout="wide")
st.title("🎨 Generative Poster (Streamlit)")

# --- session_state defaults ---
if "did_first_render" not in st.session_state:
    st.session_state.did_first_render = True
if "fixed_seed" not in st.session_state:
    st.session_state.fixed_seed = False
if "seed_value" not in st.session_state:
    st.session_state.seed_value = 123456
if "seed_randomized_once" not in st.session_state:
    st.session_state.seed_randomized_once = False

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("Controls")
    n_layers = st.slider("Layers", 3, 20, 8, 1)
    wobble = st.slider("Wobble", 0.01, 9.0, 0.18, 0.01)
    palette_mode = st.selectbox("Palette", ["pastel", "vivid", "mono", "random"])
    base_h = st.slider("Mono Hue (0~1)", 0.0, 1.0, 0.60, 0.01)

    st.divider()
    st.subheader("Output")
    save_poster = st.checkbox("Save single poster", value=True)
    save_dir = st.text_input("Save dir", "posters")
    export_fmt = st.selectbox("Export format", ["PNG", "SVG"], index=0)
    png_dpi = st.slider("PNG DPI", 72, 600, 300, 1)
    preview_width = st.slider("Preview width (px)", 600, 1900, 1200, 50)

    st.divider()
    st.subheader("Seed (재현성)")
    fixed_seed = st.checkbox("Use fixed seed", value=st.session_state.fixed_seed)
    st.session_state.fixed_seed = fixed_seed

    seed_value = st.number_input("Seed value", min_value=0, max_value=2_147_483_647,
                                 value=int(st.session_state.seed_value), step=1)
    st.session_state.seed_value = int(seed_value)

    if st.button("🎲 Randomize seed", use_container_width=True):
        new_seed = int(datetime.now().strftime("%H%M%S%f")) % 2_147_483_647
        st.session_state.seed_value = new_seed
        st.session_state.seed_randomized_once = True
        st.toast(f"New seed: {new_seed}")

    st.divider()
    st.subheader("Batch Export")
    enable_batch = st.checkbox("Enable batch ZIP export", value=False)
    batch_count = st.slider("Number of posters", 2, 50, 8, 1)
    batch_prefix = st.text_input("File prefix", "poster")
    # 고정 시드가 켜져 있으면 seed+i 방식으로 일관된 시리즈 생성
    st.caption("Tip: Use fixed seed ON → deterministic series (seed+i).")

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

def render_figure(n_layers, wobble, palette_mode, base_h, seed):
    random.seed(seed); np.random.seed(seed)
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

    ax.text(0.05, 0.95, f"Poster • {palette_mode} • seed={seed}",
            transform=ax.transAxes, fontsize=16, weight="bold", va="top")
    return fig

def generate_single(n_layers, wobble, palette_mode, base_h,
                    save, save_dir, export_fmt, png_dpi, seed=None):
    # 시드 결정
    if seed is None:
        seed = int(datetime.now().strftime("%H%M%S%f")) % 2_147_483_647

    fig = render_figure(n_layers, wobble, palette_mode, base_h, seed)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    fname_base = f"poster_{palette_mode}_seed{seed}_{timestamp}"
    saved_path = None

    if save:
        if export_fmt == "PNG":
            saved_path = os.path.join(save_dir, f"{fname_base}.png")
            fig.savefig(saved_path, dpi=png_dpi, bbox_inches="tight", pad_inches=0.0)
        else:
            saved_path = os.path.join(save_dir, f"{fname_base}.svg")
            fig.savefig(saved_path, format="svg", bbox_inches="tight", pad_inches=0.0)

    # 미리보기는 PNG 버퍼로 (SVG 선택해도 미리보기 보장)
    png_buf = io.BytesIO()
    fig.savefig(png_buf, dpi=200, format="png", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    png_buf.seek(0)

    return saved_path, seed, png_buf

def generate_batch_zip(n_layers, wobble, palette_mode, base_h,
                       export_fmt, png_dpi, count, prefix, base_seed=None):
    """count장 생성 후 ZIP(bytes) 반환. fixed_seed면 seed+i, 아니면 시간 시드 + i."""
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        tstamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        for i in range(count):
            seed = (base_seed + i) if base_seed is not None else \
                   ((int(datetime.now().strftime("%H%M%S%f")) + i) % 2_147_483_647)
            fig = render_figure(n_layers, wobble, palette_mode, base_h, seed)

            # 파일명
            if export_fmt == "PNG":
                fname = f"{prefix}_{palette_mode}_seed{seed}_{tstamp}_{i+1:02d}.png"
                buf = io.BytesIO()
                fig.savefig(buf, dpi=png_dpi, format="png", bbox_inches="tight", pad_inches=0.0)
            else:
                fname = f"{prefix}_{palette_mode}_seed{seed}_{tstamp}_{i+1:02d}.svg"
                buf = io.BytesIO()
                fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.0)
            plt.close(fig)
            buf.seek(0)
            zf.writestr(fname, buf.read())
    mem_zip.seek(0)
    return mem_zip

# --- 최초 자동 1회 ---
auto_run = st.session_state.did_first_render
st.session_state.did_first_render = False

col_btn = st.container()
with col_btn:
    left, right = st.columns([1,1], gap="small")
    with left:
        regenerate = st.button("🔁 Regenerate", use_container_width=True)
    with right:
        export_zip = st.button("📦 Export batch ZIP", use_container_width=True, disabled=not enable_batch)

img_slot = st.empty()

# ----- Actions -----
if auto_run or regenerate or st.session_state.seed_randomized_once:
    st.session_state.seed_randomized_once = False
    seed_to_use = st.session_state.seed_value if st.session_state.fixed_seed else None
    # 단일 생성
    path, used_seed, png_preview = generate_single(
        n_layers, wobble, palette_mode, base_h,
        save_poster, save_dir, export_fmt, png_dpi, seed=seed_to_use
    )
    img_slot.image(png_preview, width=preview_width)
    if save_poster:
        st.success(f"Saved: {path} (seed={used_seed})")
    else:
        st.success(f"Preview generated (seed={used_seed})")

# 배치 ZIP
if export_zip:
    base_seed = st.session_state.seed_value if st.session_state.fixed_seed else None
    zip_bytes = generate_batch_zip(
        n_layers, wobble, palette_mode, base_h, export_fmt, png_dpi,
        count=batch_count, prefix=batch_prefix, base_seed=base_seed
    )
    st.download_button(
        "⬇️ Download ZIP",
        data=zip_bytes,
        file_name=f"{batch_prefix}_{palette_mode}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip",
        mime="application/zip",
        use_container_width=True
    )
    st.toast("Batch ZIP is ready!")
