import streamlit as st
import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import matplotlib.pyplot as plt

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCB Defect Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── THEME STATE ──────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ─── THEME TOGGLE BUTTON (top-right) ──────────────────────────────────────────
_, btn_col = st.columns([8, 1])
with btn_col:
    btn_label = "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark"
    if st.button(btn_label, key="theme_btn", use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

is_dark = st.session_state.theme == "dark"

# ─── THEME PALETTES ───────────────────────────────────────────────────────────
if is_dark:
    # Deep navy + cyan — "circuit board at night"
    T = {
        "bg":               "#0a0e1a",
        "card":             "#1a2235",
        "border":           "#1e3a5f",
        "accent":           "#00d4ff",
        "accent2":          "#ff6b35",
        "green":            "#00ff88",
        "red":              "#ff3b5c",
        "text":             "#e2e8f0",
        "muted":            "#64748b",
        "header_glow":      "0 0 30px rgba(0,212,255,0.4)",
        "card_shadow":      "0 4px 24px rgba(0,0,0,0.35)",
        "uploader_bg":      "#1a2235",
        "uploader_border":  "#1e3a5f",
        "badge_defect_bg":  "rgba(255,59,92,0.15)",
        "badge_good_bg":    "rgba(0,255,136,0.10)",
        "bar_empty":        "rgba(255,255,255,0.05)",
        "footer_color":     "#334155",
        "toggle_bg":        "rgba(0,212,255,0.1)",
        "toggle_border":    "rgba(0,212,255,0.3)",
        "toggle_color":     "#00d4ff",
        "box_cv":           (0, 180, 255),
        "theme_label":      "🌙 DARK MODE  ·  Circuit board aesthetic",
        "model_dot":        "#00ff88",
    }
    # SVG bg: diagonal circuit-trace hatching
    BG_CSS = """
        background-color: #0a0e1a;
        background-image:
            radial-gradient(ellipse at 18% 48%, rgba(0,212,255,0.06) 0%, transparent 58%),
            radial-gradient(ellipse at 82% 18%, rgba(255,107,53,0.04) 0%, transparent 50%),
            url("data:image/svg+xml,%3Csvg width='80' height='80' viewBox='0 0 80 80' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none'%3E%3Cg fill='%2300d4ff' fill-opacity='0.025'%3E%3Cpath d='M0 0h40v40H0V0zm40 40h40v40H40V40zm0-40h2l-2 2V0zm0 4l4-4h2l-6 6V4zm0 4l8-8h2L40 10V8zm0 4L52 0h2L40 14v-2zm0 4L56 0h2L40 18v-2zm0 4L60 0h2L40 22v-2zm0 4L64 0h2L40 26v-2zm0 4L68 0h2L40 30v-2zm0 4L72 0h2L40 34v-2zm0 4L76 0h2L40 38v-2zm0 4L80 0v2L42 40h-2zm4 0L80 4v2L46 40h-2zm4 0L80 8v2L50 40h-2zm4 0l28-28v2L54 40h-2zm4 0l24-24v2L58 40h-2zm4 0l20-20v2L62 40h-2zm4 0l16-16v2L66 40h-2zm4 0l12-12v2L70 40h-2zm4 0l8-8v2l-6 6h-2zm4 0l4-4v2l-2 2h-2z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    """
else:
    # Crisp white + blue — "clean lab / inspection room"
    T = {
        "bg":               "#f0f4f8",
        "card":             "#ffffff",
        "border":           "#dde3ed",
        "accent":           "#0064c8",
        "accent2":          "#d4500a",
        "green":            "#00875a",
        "red":              "#c8001e",
        "text":             "#1e293b",
        "muted":            "#64748b",
        "header_glow":      "0 0 18px rgba(0,100,200,0.12)",
        "card_shadow":      "0 2px 14px rgba(0,0,0,0.07)",
        "uploader_bg":      "#ffffff",
        "uploader_border":  "#b0bec5",
        "badge_defect_bg":  "rgba(200,0,30,0.08)",
        "badge_good_bg":    "rgba(0,135,90,0.08)",
        "bar_empty":        "rgba(0,0,0,0.07)",
        "footer_color":     "#94a3b8",
        "toggle_bg":        "rgba(0,100,200,0.07)",
        "toggle_border":    "rgba(0,100,200,0.25)",
        "toggle_color":     "#0064c8",
        "box_cv":           (0, 100, 200),
        "theme_label":      "☀️ LIGHT MODE  ·  Clean lab inspection aesthetic",
        "model_dot":        "#00875a",
    }
    # SVG bg: fine crosshatch grid — like graph paper / measurement sheet
    BG_CSS = """
        background-color: #f0f4f8;
        background-image:
            radial-gradient(ellipse at 15% 10%, rgba(0,100,200,0.06) 0%, transparent 55%),
            radial-gradient(ellipse at 88% 85%, rgba(0,135,90,0.05) 0%, transparent 50%),
            url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' stroke='%230064c8' stroke-opacity='0.05' stroke-width='1'%3E%3Cpath d='M0 10h40M0 20h40M0 30h40M10 0v40M20 0v40M30 0v40'/%3E%3C/g%3E%3Cg fill='%230064c8' fill-opacity='0.06'%3E%3Ccircle cx='10' cy='10' r='1'/%3E%3Ccircle cx='20' cy='10' r='1'/%3E%3Ccircle cx='30' cy='10' r='1'/%3E%3Ccircle cx='10' cy='20' r='1'/%3E%3Ccircle cx='20' cy='20' r='1'/%3E%3Ccircle cx='30' cy='20' r='1'/%3E%3Ccircle cx='10' cy='30' r='1'/%3E%3Ccircle cx='20' cy='30' r='1'/%3E%3Ccircle cx='30' cy='30' r='1'/%3E%3C/g%3E%3C/svg%3E");
    """

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

    :root {{
        --bg:      {T['bg']};
        --card:    {T['card']};
        --border:  {T['border']};
        --accent:  {T['accent']};
        --accent2: {T['accent2']};
        --green:   {T['green']};
        --red:     {T['red']};
        --text:    {T['text']};
        --muted:   {T['muted']};
    }}

    html, body, [class*="css"] {{
        font-family: 'Exo 2', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }}

    .stApp {{
        {BG_CSS}
        transition: background-color 0.35s ease;
    }}

    /* ── TOGGLE BUTTON ── */
    div[data-testid="stButton"] button {{
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.07em !important;
        background: {T['toggle_bg']} !important;
        border: 1px solid {T['toggle_border']} !important;
        color: {T['toggle_color']} !important;
        border-radius: 8px !important;
        transition: all 0.2s !important;
    }}
    div[data-testid="stButton"] button:hover {{
        opacity: 0.8 !important;
        transform: translateY(-1px) !important;
    }}

    /* ── HEADER ── */
    .header-wrap {{
        text-align: center;
        padding: 2rem 0 1.5rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }}
    .header-wrap h1 {{
        font-family: 'Share Tech Mono', monospace;
        font-size: 2.4rem;
        letter-spacing: 0.12em;
        color: var(--accent);
        text-shadow: {T['header_glow']};
        margin: 0;
    }}
    .header-wrap p {{
        color: var(--muted);
        font-size: 0.88rem;
        letter-spacing: 0.08em;
        margin-top: 0.4rem;
        font-family: 'Share Tech Mono', monospace;
    }}
    .theme-pill {{
        display: inline-block;
        margin-top: 0.75rem;
        padding: 0.25rem 0.9rem;
        border-radius: 999px;
        border: 1px solid var(--border);
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.65rem;
        color: var(--muted);
        letter-spacing: 0.1em;
        background: {'rgba(0,212,255,0.04)' if is_dark else 'rgba(0,100,200,0.04)'};
    }}

    /* ── CARDS ── */
    .result-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: {T['card_shadow']};
        transition: background 0.35s, border-color 0.35s;
    }}
    .result-card h3 {{
        font-family: 'Share Tech Mono', monospace;
        color: var(--accent);
        font-size: 0.82rem;
        letter-spacing: 0.1em;
        margin: 0 0 1rem 0;
        text-transform: uppercase;
    }}

    /* ── DEFECT BADGE ── */
    .defect-badge {{
        display: inline-block;
        padding: 0.4rem 1.1rem;
        border-radius: 6px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.05rem;
        font-weight: bold;
        letter-spacing: 0.05em;
    }}
    .defect-badge.defect {{
        background: {T['badge_defect_bg']};
        border: 1px solid var(--red);
        color: var(--red);
    }}
    .defect-badge.good {{
        background: {T['badge_good_bg']};
        border: 1px solid var(--green);
        color: var(--green);
    }}

    /* ── CONFIDENCE BAR ── */
    .conf-label {{
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.75rem;
        color: var(--muted);
        margin-bottom: 0.3rem;
        letter-spacing: 0.05em;
    }}
    .conf-bar-bg {{
        background: {T['bar_empty']};
        border-radius: 4px;
        height: 8px;
        width: 100%;
        overflow: hidden;
        margin-bottom: 0.25rem;
    }}
    .conf-bar-fill {{
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, var(--accent), var(--accent2));
    }}
    .conf-value {{
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.4rem;
        color: var(--accent);
        font-weight: bold;
    }}

    /* ── PROB BARS ── */
    .prob-row {{
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        gap: 0.8rem;
    }}
    .prob-name {{
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.73rem;
        color: var(--muted);
        width: 130px;
        flex-shrink: 0;
    }}
    .prob-bar-bg {{
        flex: 1;
        background: {T['bar_empty']};
        border-radius: 3px;
        height: 6px;
        overflow: hidden;
    }}
    .prob-bar-fill {{ height: 100%; border-radius: 3px; }}
    .prob-pct {{
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.73rem;
        color: var(--text);
        width: 42px;
        text-align: right;
        flex-shrink: 0;
    }}

    /* ── STATS ── */
    .stats-row {{
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }}
    .stat-box {{
        flex: 1;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: {T['card_shadow']};
    }}
    .stat-box .val {{
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.55rem;
        color: var(--accent);
    }}
    .stat-box .lbl {{
        font-size: 0.7rem;
        color: var(--muted);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-top: 0.2rem;
    }}

    /* ── SECTION TITLE ── */
    .section-title {{
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.76rem;
        letter-spacing: 0.12em;
        color: var(--muted);
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }}

    /* ── UPLOADER ── */
    .stFileUploader > div {{
        background: {T['uploader_bg']} !important;
        border: 2px dashed {T['uploader_border']} !important;
        border-radius: 12px !important;
        transition: border-color 0.25s !important;
    }}
    .stFileUploader > div:hover {{
        border-color: {T['accent']} !important;
    }}
    .stFileUploader label, .stFileUploader p, .stFileUploader span {{
        color: var(--muted) !important;
    }}

    /* ── IMAGES ── */
    div[data-testid="stImage"] img {{
        border-radius: 10px;
        box-shadow: {T['card_shadow']};
    }}

    /* ── MISC ── */
    .stSpinner > div {{ border-top-color: var(--accent) !important; }}
    footer, #MainMenu, .stDeployButton {{ visibility: hidden !important; display: none !important; }}
</style>
""", unsafe_allow_html=True)


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model = load_model("pcb_defect_classifier_final.h5")
    with open("pcb_model_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, metadata


# ─── DEFECT BOXES ─────────────────────────────────────────────────────────────
def draw_defect_boxes(original_img_np, predicted_class):
    img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = img_bgr.copy()
    box_count = 0
    bx, by, bz = T["box_cv"]  # theme-aware box color

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        img_area = gray.shape[0] * gray.shape[1]
        if (w * h) > img_area * 0.6:
            continue
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (bx, by, bz), 2)
        label_text = predicted_class.replace("_", " ")
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x, y-th-8), (x+tw+6, y), (bx, by, bz), -1)
        cv2.putText(annotated, label_text, (x+3, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        box_count += 1

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), box_count


# ─── PREDICT ──────────────────────────────────────────────────────────────────
def predict(model, metadata, img_pil):
    IMG_SIZE    = metadata["img_size"]
    class_names = metadata["class_names"]
    img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE))
    img_arr     = np.array(img_resized) / 255.0
    img_input   = np.expand_dims(img_arr, axis=0)
    preds       = model.predict(img_input, verbose=0)[0]
    idx         = np.argmax(preds)
    label       = class_names[idx]
    confidence  = float(preds[idx]) * 100
    return label, confidence, preds, class_names


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-wrap">
    <h1>⬡ PCB DEFECT DETECTOR</h1>
    <p>AUTOMATED VISUAL INSPECTION SYSTEM · MOBILENETV2 · TRANSFER LEARNING</p>
    <div class="theme-pill">{T['theme_label']}</div>
</div>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
try:
    model, metadata = load_assets()
    class_names = metadata["class_names"]
    st.markdown(f"""
    <div style="text-align:center; margin-bottom:1.5rem;">
        <span style="font-family:'Share Tech Mono',monospace; font-size:0.76rem;
              color:{T['model_dot']}; letter-spacing:0.08em;">
            ● MODEL LOADED &nbsp;|&nbsp; {len(class_names)} DEFECT CLASSES
        </span>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"⚠️ Could not load model or metadata: {e}")
    st.info("Make sure `pcb_defect_classifier_final.h5` and `pcb_model_metadata.pkl` are in the same directory.")
    st.stop()

# ─── MAIN LAYOUT ──────────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<div class="section-title">Upload PCB Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop a PCB image here",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed"
    )
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        st.markdown('<div class="section-title" style="margin-top:1rem;">Original Image</div>', unsafe_allow_html=True)
        st.image(img_pil, use_container_width=True)

with col_result:
    if uploaded:
        st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)

        with st.spinner("Analyzing PCB..."):
            label, confidence, probs, class_names = predict(model, metadata, img_pil)
            img_np = np.array(img_pil)
            annotated_img, box_count = draw_defect_boxes(img_np, label)

        st.image(annotated_img, use_container_width=True,
                 caption=f"Detected regions highlighted · {box_count} area(s) flagged")

        # Stats
        st.markdown(f"""
        <div class="stats-row" style="margin-top:1rem;">
            <div class="stat-box"><div class="val">{box_count}</div><div class="lbl">Regions Flagged</div></div>
            <div class="stat-box"><div class="val">{confidence:.1f}%</div><div class="lbl">Confidence</div></div>
            <div class="stat-box"><div class="val">{img_pil.size[0]}×{img_pil.size[1]}</div><div class="lbl">Resolution</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Verdict badge + confidence bar
        is_defect = label.lower() != "good"
        badge_cls = "defect" if is_defect else "good"
        verdict   = f"{'⚠' if is_defect else '✓'}  {label.replace('_', ' ').upper()}"

        st.markdown(f"""
        <div class="result-card">
            <h3>// Detection Result</h3>
            <div style="margin-bottom:1rem;">
                <span class="defect-badge {badge_cls}">{verdict}</span>
            </div>
            <div class="conf-label">CONFIDENCE SCORE</div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{confidence:.1f}%"></div>
            </div>
            <div class="conf-value">{confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        colors = [T['accent'], T['accent2'], T['green'], T['red'], "#a78bfa", "#fb923c"]
        sorted_idx = np.argsort(probs)[::-1]
        top_idx = int(np.argmax(probs))
        prob_html = ""
        for rank, i in enumerate(sorted_idx):
            name  = class_names[i].replace("_", " ")
            pct   = probs[i] * 100
            col   = colors[rank % len(colors)]
            bold  = f"color:{T['text']}; font-weight:600;" if i == top_idx else ""
            prob_html += f"""
            <div class="prob-row">
              <div class="prob-name" style="{bold}">{name}</div>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{pct:.1f}%; background:{col};"></div>
              </div>
              <div class="prob-pct" style="{bold}">{pct:.1f}%</div>
            </div>"""

        st.markdown(f"""
        <div class="result-card">
            <h3>// Class Probabilities</h3>
            {prob_html}
        </div>
        """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; margin-top:3rem; padding-top:1rem;
            border-top:1px solid {T['border']};">
    <span style="font-family:'Share Tech Mono',monospace; font-size:0.7rem;
          color:{T['footer_color']}; letter-spacing:0.08em;">
        PCB DEFECT DETECTION · MOBILENETV2 TRANSFER LEARNING · {'🌙 DARK MODE' if is_dark else '☀️ LIGHT MODE'}
    </span>
</div>
""", unsafe_allow_html=True)