# streamlit_app.py — One-Click Poster
# (+ Seed, + Batch ZIP, + PNG/SVG & DPI, + 🎨 Custom Palette Editor, + ✒ Typography Overlay)
import os, io, math, random, zipfile, tempfile
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from matplotlib.colors import hsv_to_rgb
from matplotlib.font_manager import FontProperties, fontManager
import matplotlib.transforms as transforms
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch

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
if "uploaded_font_path" not in st.session_state:
    st.session_state.uploaded_font_path = None

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

    st.divider()
    st.subheader("✒ Typography Overlay")
    enable_typo = st.checkbox("Enable text overlay", value=False)
    typo_text = st.text_area("Text", "HELLO, POSTER\nSubheading here", height=100)
    col_pos = st.columns(2)
    with col_pos[0]:
        pos_x = st.slider("X position", 0.0, 1.0, 0.08, 0.005)
    with col_pos[1]:
        pos_y = st.slider("Y position", 0.0, 1.0, 0.88, 0.005)
    align = st.selectbox("Horizontal align", ["left", "center", "right"], index=0)
    valign = st.selectbox("Vertical align", ["top", "center", "bottom"], index=0)
    font_size = st.slider("Font size (pt)", 6, 180, 72, 1)
    line_spacing = st.slider("Line spacing (×)", 0.6, 2.5, 1.2, 0.05)
    tracking_em = st.slider("Tracking (em)", -0.1, 0.5, 0.05, 0.01)
    col_fs = st.columns(2)
    with col_fs[0]:
        font_weight = st.selectbox("Weight", ["normal", "bold"], index=1)
    with col_fs[1]:
        font_style = st.selectbox("Style", ["normal", "italic"], index=0)

    typo_color = st.color_picker("Text color", "#111111")
    typo_alpha = st.slider("Opacity", 0.0, 1.0, 0.95, 0.01)

    st.caption("Font: Pick installed family or upload a TTF/OTF.")
    font_family = st.selectbox(
        "Font family",
        ["Default", "DejaVu Sans", "Arial", "Times New Roman", "NanumGothic", "Uploaded font"],
        index=0
    )
    up_font = st.file_uploader("Upload TTF/OTF", type=["ttf", "otf"])
    if up_font is not None:
        try:
            # Save to temp file (persistent per run)
            tmp_dir = tempfile.gettempdir()
            fpath = os.path.join(tmp_dir, f"uploaded_{datetime.now().strftime('%H%M%S%f')}_{up_font.name}")
            with open(fpath, "wb") as f:
                f.write(up_font.read())
            fontManager.addfont(fpath)
            st.session_state.uploaded_font_path = fpath
            st.success("Font registered!")
        except Exception as e:
            st.error(f"Font load error: {e}")

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

# -------- Typography drawing (letter-spacing & line-height) --------
def build_fontprop(font_family, weight, style):
    """Return a Matplotlib FontProperties object resolving uploaded font if chosen."""
    if font_family == "Uploaded font" and st.session_state.uploaded_font_path:
        return FontProperties(fname=st.session_state.uploaded_font_path,
                              weight=weight, style=style)
    elif font_family == "Default":
        return FontProperties(weight=weight, style=style)
    else:
        return FontProperties(family=font_family, weight=weight, style=style)

def draw_text_with_typography(ax, text, x_axes, y_axes,
                              font_prop: FontProperties,
                              size_pt: float,
                              color=(0,0,0), alpha=1.0,
                              ha="left", va="top",
                              line_spacing=1.2,
                              tracking_em=0.0):
    """
    Draw multiline text in Axes coords with:
      - custom tracking (em)
      - custom line spacing (× font size)
      - alignment (left/center/right, top/center/bottom)
    Uses per-character TextPath to compute accurate advances.
    """
    fig = ax.figure
    # Normalize alignment keys
    ha = {"left":"left","center":"center","right":"right"}.get(ha, "left")
    va = {"top":"top","center":"center","bottom":"bottom"}.get(va, "top")

    lines = str(text).split("\n")

    # Measure each line total width in points using TextPath
    line_width_pts = []
    glyph_widths_pts = []  # list of lists per line
    for ln in lines:
        widths = []
        total = 0.0
        for ch in ln:
            tp = TextPath((0,0), ch if ch != " " else " ", size=size_pt, prop=font_prop, usetex=False)
            w = tp.get_extents().width  # in points
            # add tracking per character (em * size_pt)
            widths.append(w)
            total += w + tracking_em * size_pt
        if len(ln) > 0:
            total -= tracking_em * size_pt  # no trailing tracking after last char
        line_width_pts.append(total)
        glyph_widths_pts.append(widths)

    # Compute baseline y offsets for vertical alignment in points
    line_height_pts = size_pt * line_spacing
    total_height_pts = line_height_pts * max(1, len(lines))
    if va == "top":
        first_line_y_offset_pts = 0.0
    elif va == "center":
        first_line_y_offset_pts = + (total_height_pts/2 - line_height_pts)  # move down half height minus 1 line
    else:  # bottom
        first_line_y_offset_pts = + (total_height_pts - line_height_pts)

    # Draw each line by per-character placement using ScaledTranslation in points
    for li, ln in enumerate(lines):
        # horizontal base offset per alignment
        lw = line_width_pts[li]
        if ha == "left":
            x_offset_pts = 0.0
        elif ha == "center":
            x_offset_pts = - lw / 2.0
        else:  # right
            x_offset_pts = - lw

        # vertical offset for this line (downward is negative in axes; we handle by negative points)
        y_offset_pts = - (first_line_y_offset_pts - li * line_height_pts)

        # advance cursor for this line
        cursor_x_pts = x_offset_pts

        if ln == "":
            continue

        for ci, ch in enumerate(ln):
            # place each glyph via Axes transform + ScaledTranslation
            this_T = ax.transAxes + transforms.ScaledTranslation(cursor_x_pts/72.0, y_offset_pts/72.0, fig.dpi_scale_trans)
            # draw as simple text (lets Matplotlib handle hinting/kern)
            ax.text(x_axes, y_axes, ch, transform=this_T,
                    fontproperties=font_prop, fontsize=size_pt,
                    color=color, alpha=alpha,
                    ha="left", va="top")  # we already handle alignment
            # advance cursor by glyph width + tracking
            g_w = glyph_widths_pts[li][ci]
            cursor_x_pts += g_w + tracking_em * size_pt

# -------------- Figure renderer with optional overlay --------------
def render_figure(n_layers, wobble, palette_mode, base_h, seed, palette_override=None,
                  typo_cfg=None):
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

    # Title stamp
    ax.text(0.05, 0.95,
            f"Poster • {'custom' if (palette_override and len(palette_override)>=2) else palette_mode} • seed={seed}",
            transform=ax.transAxes, fontsize=16, weight="bold", va="top")

    # Typography overlay
    if typo_cfg and typo_cfg.get("enable"):
        try:
            fp = build_fontprop(typo_cfg["family"], typo_cfg["weight"], typo_cfg["style"])
            draw_text_with_typography(
                ax,
                text=typo_cfg["text"],
                x_axes=typo_cfg["x"],
                y_axes=typo_cfg["y"],
                font_prop=fp,
                size_pt=typo_cfg["size_pt"],
                color=typo_cfg["color"],
                alpha=typo_cfg["alpha"],
                ha=typo_cfg["ha"],
                va=typo_cfg["va"],
                line_spacing=typo_cfg["line_spacing"],
                tracking_em=typo_cfg["tracking_em"],
            )
        except Exception as e:
            # 안전 장치: 폴백 텍스트
            ax.text(0.5, 0.5, f"[Typography error]\n{e}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="red")

    return fig

def generate_single(n_layers, wobble, palette_mode, base_h,
                    save, save_dir, export_fmt, png_dpi, seed=None,
                    palette_override=None, typo_cfg=None):
    if seed is None:
        seed = int(datetime.now().strftime("%H%M%S%f")) % 2_147_483_647

    fig = render_figure(n_layers, wobble, palette_mode, base_h, seed,
                        palette_override=palette_override, typo_cfg=typo_cfg)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    name_mode = 'custom' if (palette_override and len(palette_override)>=2) else palette_mode
    fname_base = f"poster_{name_mode}_seed{seed}_{timestamp}"
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
                       export_fmt, png_dpi, count, prefix, base_seed=None,
                       palette_override=None, typo_cfg=None):
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        tstamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        for i in range(count):
            seed = (base_seed + i) if base_seed is not None else \
                   ((int(datetime.now().strftime("%H%M%S%f")) + i) % 2_147_483_647)
            fig = render_figure(n_layers, wobble, palette_mode, base_h, seed,
                                palette_override=palette_override, typo_cfg=typo_cfg)

            name_mode = 'custom' if (palette_override and len(palette_override)>=2) else palette_mode
            if export_fmt == "PNG":
                fname = f"{prefix}_{name_mode}_seed{seed}_{tstamp}_{i+1:02d}.png"
                buf = io.BytesIO()
                fig.savefig(buf, dpi=png_dpi, format="png", bbox_inches="tight", pad_inches=0.0)
            else:
                fname = f"{prefix}_{name_mode}_seed{seed}_{tstamp}_{i+1:02d}.svg"
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

# ----- Assemble typography config -----
typo_cfg = None
if enable_typo:
    # Resolve color to RGB tuple (0-1)
    tc = hex_to_rgb01(typo_color)
    typo_cfg = {
        "enable": True,
        "text": typo_text,
        "x": pos_x, "y": pos_y,
        "family": font_family,
        "weight": font_weight,
        "style": font_style,
        "size_pt": float(font_size),
        "line_spacing": float(line_spacing),
        "tracking_em": float(tracking_em),
        "color": tc,
        "alpha": float(typo_alpha),
        "ha": align,
        "va": valign,
    }

# ----- Actions -----
if auto_run or regenerate or st.session_state.seed_randomized_once:
    st.session_state.seed_randomized_once = False
    seed_to_use = st.session_state.seed_value if st.session_state.fixed_seed else None
    path, used_seed, png_preview = generate_single(
        n_layers, wobble, palette_mode, base_h,
        save_poster, save_dir, export_fmt, png_dpi,
        seed=seed_to_use,
        palette_override=custom_palette if use_custom_palette else None,
        typo_cfg=typo_cfg
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
        palette_override=custom_palette if use_custom_palette else None,
        typo_cfg=typo_cfg
    )
    st.download_button(
        "⬇️ Download ZIP",
        data=zip_bytes,
        file_name=f"{batch_prefix}_{'custom' if (use_custom_palette and custom_palette) else palette_mode}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip",
        mime="application/zip",
        use_container_width=True
    )
    st.toast("Batch ZIP is ready!")
