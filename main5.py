# streamlit_app.py — One-Click Poster
# (+ Seed, + Batch ZIP, + PNG/SVG & DPI, + 🎨 Custom Palette Editor)
import os, io, math, random, zipfile
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
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
if "manual_hex" not in st.session_state:
    st.session_state.manual_hex = ["#5ec1ff", "#ffd166", "#ef476f", "#06d6a0", "#8338ec", "#118ab2"]

# ----------------- Utility: Palette -----------------
def hex_to_rgb01(hex_str: str):
    """#RRGGBB -> (r,g,b) in [0,1]"""
    h = hex_str.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid HEX: {hex_str}")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b)

def rgb_any_to01(r, g, b):
    """Accept 0-255 ints or 0-1 floats; normalize to [0,1]."""
    vals = []
    for v in (r, g, b):
        try:
            v = float(v)
        except:
            v = 0.0
        if v > 1.0:  # assume 0-255
            v = v / 255.0
        vals.append(max(0.0, min(1.0, v)))
    return tuple(vals)

def parse_palette_csv(file_bytes) -> list:
    """CSV columns: either hex or r,g,b (0-1 or 0-255). Returns [(r,g,b), ...] in [0,1]."""
    df = pd.read_csv(file_bytes)
    cols = [c.lower() for c in df.columns]
    palette = []
    if "hex" in cols:
        for hx in df["hex"]:
            palette.append(hex_to_rgb01(str(hx)))
    elif all(c in cols for c in ["r", "g", "b"]):
        for _, row in df.iterrows():
            palette.append(rgb_any_to01(row["r"], row["g"], row["b"]))
    else:
        raise ValueError("CSV must include 'hex' or 'r,g,b' columns.")
    # 제한적 중복 제거
    uniq = []
    seen = set()
    for rgb in palette:
        key = tuple(round(x, 4) for x in rgb)
        if key not in seen:
            seen.add(key)
            uniq.append(rgb)
    return uniq[:24]  # 안전하게 상한

def palette_to_csv_bytes(hex_list: list):
    rows = []
    for hx in hex_list:
        r, g, b = hex_to_rgb01(hx)
        rows.append({"hex": hx, "r": round(r, 4), "g": round(g, 4), "b": round(b, 4)})
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("Controls")
    n_layers = st.slider("Layers", 3, 20, 8, 1)
    wobble = st.slider("Wobble", 0.01, 9.0, 0.18, 0.01)
    palette_mode = st.selectbox("Palette (auto)", ["pastel", "vivid", "mono", "random"])
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
    st.caption("Tip: Use fixed seed ON → deterministic series (seed+i).")

    st.divider()
    st.subheader("🎨 Custom Palette Editor")
    use_custom_palette = st.checkbox("Use custom palette (override auto palette)", value=False)
    pal_source = st.radio("Source", ["Manual HEX", "Upload CSV"], horizontal=True)

    custom_palette = None
    if use_custom_palette:
        if pal_source == "Upload CSV":
            up = st.file_uploader("Upload palette CSV (hex or r,g,b)", type=["csv"])
            if up is not None:
                try:
                    custom_palette = parse_palette_csv(up)
                    st.success(f"Loaded {len(custom_palette)} colors from CSV.")
                except Exception as e:
                    st.error(str(e))
        else:
            st.caption("Pick colors (up to 12). CSV로 저장 후 재사용 가능.")
            n_hex = st.slider("Number of colors", 2, 12, len(st.session_state.manual_hex), 1)
            # 확장/축소
            cur = st.session_state.manual_hex
            if n_hex > len(cur):
                cur = cur + ["#888888"] * (n_hex - len(cur))
            else:
                cur = cur[:n_hex]
            cols = st.columns(3)
            for i in range(n_hex):
                with cols[i % 3]:
                    cur[i] = st.color_picker(f"Color {i+1}", value=cur[i])
            st.session_state.manual_hex = cur
            # 프리뷰
            st.write("Preview:")
            row = st.columns(len(cur))
            for i, c in enumerate(cur):
                with row[i]:
                    st.markdown(f"<div style='width:100%;height:24px;border-radius:6px;background:{c}'></div>", unsafe_allow_html=True)
                    st.caption(c)
            # CSV로 저장
            csv_bytes = palette_to_csv_bytes(cur)
            st.download_button("⬇️ Download palette.csv", data=csv_bytes,
                               file_name="palette.csv", mime="text/csv", use_container_width=True)
            # 내부 팔레트로 변환
            try:
                custom_palette = [hex_to_rgb01(hx) for hx in cur]
            except Exception as e:
                st.error(f"HEX parse error: {e}")

# ----------------- Core rendering -----------------
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

def render_figure(n_layers, wobble, palette_mode, base_h, seed, palette_override=None):
    random.seed(seed); np.random.seed(seed)
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.axis("off")
    ax.set_facecolor((0.97,0.97,0.97))

    # 팔레트 결정: custom > auto
    if palette_override and len(palette_override) >= 2:
        palette = palette_override
    else:
        palette = make_palette(6, mode=palette_mode, base_h=base_h)

    for _ in range(n_layers):
        cx, cy = random.random(), random.random()
        rr = random.uniform(0.15, 0.45)
        x, y = blob((cx,cy), r=rr, wobble=wobble)
        color = random.choice(palette)
        alpha = random.uniform(0.3, 0.6)
        ax.fill(x, y, color=color, alpha=alpha, edgecolor=(0,0,0,0))

    ax.text(0.05, 0.95,
            f"Poster • {'custom' if (palette_override and len(palette_override)>=2) else palette_mode} • seed={seed}",
            transform=ax.transAxes, fontsize=16, weight="bold", va="top")
    return fig

def generate_single(n_layers, wobble, palette_mode, base_h,
                    save, save_dir, export_fmt, png_dpi, seed=None, palette_override=None):
    if seed is None:
        seed = int(datetime.now().strftime("%H%M%S%f")) % 2_147_483_647

    fig = render_figure(n_layers, wobble, palette_mode, base_h, seed, palette_override=palette_override)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    fname_base = f"poster_{'custom' if (palette_override and len(palette_override)>=2) else palette_mode}_seed{seed}_{timestamp}"
    saved_path = None

    if save:
        if export_fmt == "PNG":
            saved_path = os.path.join(save_dir, f"{fname_base}.png")
            fig.savefig(saved_path, dpi=png_dpi, bbox_inches="tight", pad_inches=0.0)
        else:
            saved_path = os.path.join(save_dir, f"{fname_base}.svg")
            fig.savefig(saved_path, format="svg", bbox_inches="tight", pad_inches=0.0)

    png_buf = io.BytesIO()
    fig.savefig(png_buf, dpi=200, format="png", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    png_buf.seek(0)
    return saved_path, seed, png_buf

def generate_batch_zip(n_layers, wobble, palette_mode, base_h,
                       export_fmt, png_dpi, count, prefix, base_seed=None, palette_override=None):
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        tstamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        for i in range(count):
            seed = (base_seed + i) if base_seed is not None else \
                   ((int(datetime.now().strftime("%H%M%S%f")) + i) % 2_147_483_647)
            fig = render_figure(n_layers, wobble, palette_mode, base_h, seed, palette_override=palette_override)

            if export_fmt == "PNG":
                fname = f"{prefix}_{'custom' if (palette_override and len(palette_override)>=2) else palette_mode}_seed{seed}_{tstamp}_{i+1:02d}.png"
                buf = io.BytesIO()
                fig.savefig(buf, dpi=png_dpi, format="png", bbox_inches="tight", pad_inches=0.0)
            else:
                fname = f"{prefix}_{'custom' if (palette_override and len(palette_override)>=2) else palette_mode}_seed{seed}_{tstamp}_{i+1:02d}.svg"
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
    path, used_seed, png_preview = generate_single(
        n_layers, wobble, palette_mode, base_h,
        save_poster, save_dir, export_fmt, png_dpi,
        seed=seed_to_use, palette_override=custom_palette if use_custom_palette else None
    )
    img_slot.image(png_preview, width=preview_width)
    if save_poster:
        st.success(f"Saved: {path} (seed={used_seed})")
    else:
        st.success(f"Preview generated (seed={used_seed})")

if export_zip:
    base_seed = st.session_state.seed_value if st.session_state.fixed_seed else None
    zip_bytes = generate_batch_zip(
        n_layers, wobble, palette_mode, base_h, export_fmt, png_dpi,
        count=batch_count, prefix=batch_prefix, base_seed=base_seed,
        palette_override=custom_palette if use_custom_palette else None
    )
    st.download_button(
        "⬇️ Download ZIP",
        data=zip_bytes,
        file_name=f"{batch_prefix}_{'custom' if (use_custom_palette and custom_palette) else palette_mode}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip",
        mime="application/zip",
        use_container_width=True
    )
    st.toast("Batch ZIP is ready!")
