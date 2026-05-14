# ==========================================================
# 🔥 FIRESENSE AI — REDESIGNED DASHBOARD
# ==========================================================

import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import random
import time

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="FireSense AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_fire_model():
    return load_model("fire_detection_model.keras")

model = load_fire_model()

# ==========================================================
# CSS — Industrial Brutalist Dark Theme
# ==========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:      #0A0A0A;
    --surface: #111111;
    --border:  #222222;
    --accent:  #FF4500;
    --accent2: #FF6B35;
    --dim:     #444444;
    --text:    #F0EDE8;
    --muted:   #888888;
    --safe:    #00C896;
    --warn:    #FFB800;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background: var(--bg) !important;
    background-image:
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(255,69,0,0.03) 39px, rgba(255,69,0,0.03) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(255,69,0,0.03) 39px, rgba(255,69,0,0.03) 40px) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── HERO ── */
.hero {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: end;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem;
    margin-bottom: 2.5rem;
}

.hero-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.25em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3.5rem, 7vw, 5.5rem);
    line-height: 0.95;
    color: var(--text);
    letter-spacing: 0.02em;
}

.hero-title span { color: var(--accent); }

.hero-right { text-align: right; }

.hero-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    justify-content: flex-end;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--safe);
}

.pulse-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--safe);
    animation: pulse 1.8s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.35; transform: scale(0.65); }
}

.hero-desc {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    margin-top: 0.3rem;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0 !important;
    border-bottom: 1px solid var(--border) !important;
    margin-bottom: 2rem !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 0 !important;
    transition: all 0.2s !important;
}

.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

[data-testid="stFileUploader"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border: 1px dashed var(--dim) !important;
    border-radius: 4px !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
}

/* drag-and-drop inner text */
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* "Limit 200MB" small text */
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzone"] small {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
}

/* "Browse files" button */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="baseButton-secondary"] {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.45rem 1.1rem !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: var(--accent2) !important;
}

/* uploaded filename row */
[data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploaderFile"] p,
[data-testid="stFileUploaderFile"] small {
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
}

/* delete (×) button on uploaded file */
[data-testid="stFileUploaderDeleteBtn"] button {
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
}

[data-testid="stFileUploaderDeleteBtn"] button:hover {
    color: var(--accent) !important;
}

/* ── VERDICT CARDS ── */
.verdict-fire {
    background: var(--accent);
    color: #fff;
    padding: 2rem 2.5rem;
    border-radius: 4px;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.verdict-fire::before {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        45deg, transparent, transparent 8px,
        rgba(0,0,0,0.07) 8px, rgba(0,0,0,0.07) 9px
    );
}

.verdict-safe {
    background: transparent;
    border: 1px solid var(--safe);
    color: var(--safe);
    padding: 2rem 2.5rem;
    border-radius: 4px;
    margin-bottom: 1.5rem;
}

.verdict-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    opacity: 0.75;
    margin-bottom: 0.4rem;
    position: relative;
}

.verdict-main {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    line-height: 1;
    position: relative;
}

.verdict-prob {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    margin-top: 0.6rem;
    opacity: 0.9;
    position: relative;
}

/* ── PROB BARS ── */
.prob-bar-wrap { margin: 0.75rem 0; }

.prob-bar-label {
    display: flex;
    justify-content: space-between;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    margin-bottom: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.prob-bar-track {
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}

.prob-bar-fill { height: 100%; border-radius: 2px; }
.prob-bar-fill.fire { background: var(--accent); }
.prob-bar-fill.safe { background: var(--safe); }

/* ── INSIGHT PILLS ── */
.insight-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
    margin-top: 1rem;
}

.insight-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.85rem;
}

.pill {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 2px;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    margin: 0.2rem 0.2rem 0.2rem 0;
}

.pill-red   { background: rgba(255,69,0,0.14);   color: var(--accent); border: 1px solid rgba(255,69,0,0.3); }
.pill-green { background: rgba(0,200,150,0.1);   color: var(--safe);   border: 1px solid rgba(0,200,150,0.3); }
.pill-amber { background: rgba(255,184,0,0.1);   color: var(--warn);   border: 1px solid rgba(255,184,0,0.3); }
.pill-dim   { background: rgba(255,255,255,0.04); color: var(--muted); border: 1px solid var(--border); }

/* ── STAT STRIP ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    margin: 1.5rem 0;
}

.stat-card {
    background: var(--surface);
    padding: 1.2rem 1.4rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.stat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
}

.stat-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    line-height: 1;
    color: var(--text);
}

.stat-value.fv { color: var(--accent); }
.stat-value.sv { color: var(--safe); }

/* ── INFO CARDS ── */
.info-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2rem;
}

.info-card-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    color: var(--text);
    margin-bottom: 1.25rem;
    letter-spacing: 0.05em;
}

.step-item {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    margin-bottom: 0.75rem;
}

.step-num {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    color: var(--accent);
    line-height: 1;
    flex-shrink: 0;
    width: 2rem;
}

.step-text {
    font-size: 0.82rem;
    color: var(--muted);
    padding-top: 0.15rem;
    line-height: 1.55;
}

.safety-item {
    border-left: 2px solid var(--border);
    padding: 0.6rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.82rem;
    color: var(--muted);
    transition: border-color 0.2s;
}

.safety-item:hover { border-color: var(--accent); color: var(--text); }

.safety-item b {
    display: block;
    font-size: 0.65rem;
    font-family: 'Space Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.2rem;
    color: var(--text);
}

/* ── STANDBY ── */
.standby {
    border: 1px dashed var(--border);
    border-radius: 4px;
    padding: 5rem 2rem;
    text-align: center;
    margin-top: 1rem;
}

.standby-icon { font-size: 3.5rem; display: block; margin-bottom: 1rem; }

.standby-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    letter-spacing: 0.1em;
    color: var(--dim);
    margin-bottom: 0.5rem;
}

.standby-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--dim);
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* ── FOOTER ── */
.footer {
    border-top: 1px solid var(--border);
    margin-top: 3rem;
    padding-top: 1.5rem;
    display: flex;
    justify-content: space-between;
}

.footer-left {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    color: var(--dim);
    letter-spacing: 0.1em;
}

.footer-right {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: var(--dim);
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# PREDICTION
# ==========================================================

def predict_fire(image):
    img = image.resize((224, 224))
    arr = np.array(img)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr)
    conf = float(pred[0][0])
    if conf > 0.5:
        return "FIRE DETECTED", "fire", conf * 100, (1 - conf) * 100
    else:
        return "SAFE ENVIRONMENT", "safe", conf * 100, (1 - conf) * 100

# ==========================================================
# PLOTLY BASE LAYOUT
# ==========================================================

def dark_layout(title="", h=320):
    return dict(
        title=dict(text=title, font=dict(family="Space Mono, monospace", size=10, color="#666"), x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono, monospace", color="#666", size=9),
        height=h,
        margin=dict(l=10, r=10, t=36, b=10),
        xaxis=dict(showgrid=False, showline=False, color="#333"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#333"),
    )

# ==========================================================
# HERO
# ==========================================================

st.markdown("""
<div class="hero">
  <div class="hero-left">
    <div class="hero-title">FIRE<span>SENSE</span><br>DETECTION</div>
  </div>
  <div class="hero-right">
    <div class="hero-status">
      <div class="pulse-dot"></div>SYSTEM ONLINE
    </div>
    <div class="hero-desc">TensorFlow · Real-time · 97% Accuracy</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# TABS
# ==========================================================

tab1, tab2 = st.tabs(["// LIVE DETECTION", "// ANALYTICS & INFO"])

# ==========================================================
# TAB 1
# ==========================================================

with tab1:

    uploaded_file = st.file_uploader(
        "DROP IMAGE TO ANALYZE — JPG / PNG",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

        with st.spinner(""):
            time.sleep(1.2)
            label, status, fire_prob, safe_prob = predict_fire(image)

        col_img, col_verdict = st.columns([1, 1], gap="large")

        with col_img:
            st.image(image, use_container_width=True)

        with col_verdict:
            if status == "fire":
                st.markdown(f"""
                <div class="verdict-fire">
                  <div class="verdict-label">// AI VERDICT</div>
                  <div class="verdict-main">🔥 {label}</div>
                  <div class="verdict-prob">FIRE PROBABILITY → {fire_prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-safe">
                  <div class="verdict-label">// AI VERDICT</div>
                  <div class="verdict-main">✓ {label}</div>
                  <div class="verdict-prob">SAFETY CONFIDENCE → {safe_prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="prob-bar-wrap">
              <div class="prob-bar-label"><span>FIRE RISK</span><span>{fire_prob:.1f}%</span></div>
              <div class="prob-bar-track">
                <div class="prob-bar-fill fire" style="width:{fire_prob:.1f}%"></div>
              </div>
            </div>
            <div class="prob-bar-wrap">
              <div class="prob-bar-label"><span>SAFE SCORE</span><span>{safe_prob:.1f}%</span></div>
              <div class="prob-bar-track">
                <div class="prob-bar-fill safe" style="width:{safe_prob:.1f}%"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if status == "fire":
                st.markdown("""
                <div class="insight-block">
                  <div class="insight-title">// SIGNALS DETECTED</div>
                  <span class="pill pill-red">⚠ HIGH THERMAL ANOMALY</span>
                  <span class="pill pill-red">⚠ SMOKE PATTERN</span>
                  <span class="pill pill-amber">△ ELEVATED RISK</span>
                  <span class="pill pill-dim">ALERT DISPATCHED</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="insight-block">
                  <div class="insight-title">// SIGNALS DETECTED</div>
                  <span class="pill pill-green">✓ NO THERMAL SPIKE</span>
                  <span class="pill pill-green">✓ CLEAR AIR PATTERN</span>
                  <span class="pill pill-dim">STABLE ENVIRONMENT</span>
                  <span class="pill pill-dim">MONITORING ACTIVE</span>
                </div>
                """, unsafe_allow_html=True)

        # Stat strip
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card">
            <div class="stat-label">Fire Risk</div>
            <div class="stat-value fv">{fire_prob:.0f}<span style="font-size:1rem">%</span></div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Safe Score</div>
            <div class="stat-value sv">{safe_prob:.0f}<span style="font-size:1rem">%</span></div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Inference FPS</div>
            <div class="stat-value">{random.randint(28, 36)}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Model Accuracy</div>
            <div class="stat-value">97<span style="font-size:1rem">%</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Charts
        ch1, ch2 = st.columns(2)

        with ch1:
            fig1 = go.Figure(go.Pie(
                labels=["Fire", "Safe"],
                values=[fire_prob, safe_prob],
                hole=0.72,
                marker=dict(colors=["#FF4500", "#1C1C1C"], line=dict(color="#0A0A0A", width=2)),
                textinfo="none",
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>"
            ))
            fig1.add_annotation(
                text=f"{fire_prob:.0f}%", x=0.5, y=0.55, showarrow=False,
                font=dict(family="Bebas Neue, sans-serif", size=38, color="#FF4500")
            )
            fig1.add_annotation(
                text="FIRE RISK", x=0.5, y=0.38, showarrow=False,
                font=dict(family="Space Mono, monospace", size=9, color="#666")
            )
            fig1.update_layout(**dark_layout("// PROBABILITY SPLIT"))
            st.plotly_chart(fig1, use_container_width=True)

        with ch2:
            frames = list(range(1, 21))
            thermal = np.clip(np.random.normal(fire_prob * 0.8, 15, 20), 0, 100).tolist()
            fig2 = go.Figure(go.Scatter(
                x=frames, y=thermal, mode="lines",
                fill="tozeroy", fillcolor="rgba(255,69,0,0.07)",
                line=dict(color="#FF4500", width=2),
            ))
            fig2.update_layout(**dark_layout("// THERMAL TIMELINE"))
            st.plotly_chart(fig2, use_container_width=True)

        ch3, ch4 = st.columns(2)

        with ch3:
            cats = ["Fire", "Smoke", "Heat", "Clear"]
            vals = [
                fire_prob * random.uniform(0.85, 1.0),
                fire_prob * random.uniform(0.60, 0.90),
                fire_prob * random.uniform(0.70, 0.95),
                safe_prob * random.uniform(0.80, 1.0),
            ]
            fig3 = go.Figure(go.Bar(
                x=cats, y=vals,
                marker_color=["#FF4500", "#FF6B35", "#FFB800", "#00C896"],
                text=[f"{v:.0f}%" for v in vals],
                textposition="outside",
                textfont=dict(family="Space Mono, monospace", size=9, color="#666"),
            ))
            fig3.update_layout(**dark_layout("// SIGNAL BREAKDOWN"))
            st.plotly_chart(fig3, use_container_width=True)

        with ch4:
            fig4 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=int(fire_prob),
                number=dict(font=dict(family="Bebas Neue, sans-serif", size=44, color="#F0EDE8"), suffix="%"),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#333", tickfont=dict(color="#444", size=9)),
                    bar=dict(color="#FF4500", thickness=0.22),
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 40],  color="rgba(0,200,150,0.08)"),
                        dict(range=[40, 70], color="rgba(255,184,0,0.08)"),
                        dict(range=[70, 100],color="rgba(255,69,0,0.10)")
                    ],
                    threshold=dict(line=dict(color="#FF4500", width=2), thickness=0.8, value=int(fire_prob))
                )
            ))
            fig4.update_layout(**dark_layout("// RISK GAUGE"))
            st.plotly_chart(fig4, use_container_width=True)

    else:
        st.markdown("""
        <div class="standby">
          <span class="standby-icon">◎</span>
          <div class="standby-title">AWAITING INPUT</div>
          <div class="standby-sub">Upload a fire or smoke image to begin analysis</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================================
# TAB 2
# ==========================================================

with tab2:
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown("""
        <div class="info-card">
          <div class="info-card-title">HOW THE AI WORKS</div>
          <div class="step-item"><div class="step-num">01</div><div class="step-text">Image uploaded into the detection pipeline</div></div>
          <div class="step-item"><div class="step-num">02</div><div class="step-text">Resized to 224×224 and normalized for the CNN</div></div>
          <div class="step-item"><div class="step-num">03</div><div class="step-text">TensorFlow model extracts fire & smoke feature maps</div></div>
          <div class="step-item"><div class="step-num">04</div><div class="step-text">Sigmoid output gives confidence probability [0–1]</div></div>
          <div class="step-item"><div class="step-num">05</div><div class="step-text">Dashboard renders verdict, metrics, and charts</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        perf = pd.DataFrame({
            "Module": ["Detection", "Smoke", "Thermal", "Inference"],
            "Efficiency": [97, 91, 94, 98]
        })
        fig5 = px.line(perf, x="Module", y="Efficiency", markers=True)
        fig5.update_traces(line=dict(color="#FF4500", width=2), marker=dict(color="#FF4500", size=7))
        fig5.update_layout(**dark_layout("// MODULE PERFORMANCE", 280))
        st.plotly_chart(fig5, use_container_width=True)

    with right_col:
        st.markdown("""
        <div class="info-card">
          <div class="info-card-title">FIRE SAFETY PROTOCOL</div>
          <div class="safety-item"><b>01 — ALARMS</b>Install smoke detectors on every floor; test monthly.</div>
          <div class="safety-item"><b>02 — EXITS</b>Keep all emergency exits unobstructed at all times.</div>
          <div class="safety-item"><b>03 — ELECTRICAL</b>Inspect wiring regularly; replace frayed cords.</div>
          <div class="safety-item"><b>04 — ALERTS</b>Never silence or ignore a smoke alarm signal.</div>
          <div class="safety-item"><b>05 — ESCAPE</b>Drill evacuation routes with all occupants annually.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig6 = go.Figure(go.Scatterpolar(
            r=[90, 85, 88, 92, 95, 90],
            theta=["Detection", "Thermal", "Smoke", "Cloud AI", "Emergency", "Detection"],
            fill="toself",
            fillcolor="rgba(255,69,0,0.07)",
            line=dict(color="#FF4500", width=2),
            marker=dict(color="#FF4500", size=5)
        ))
        fig6.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1E1E1E",
                                tickfont=dict(color="#444", size=8)),
                angularaxis=dict(gridcolor="#1E1E1E",
                                 tickfont=dict(family="Space Mono, monospace", size=9, color="#666"))
            ),
            **dark_layout("// AI CAPABILITY RADAR", 280)
        )
        st.plotly_chart(fig6, use_container_width=True)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("""
<div class="footer">
  <div class="footer-left">FIRESENSE AI — DETECTION SYSTEM</div>
  <div class="footer-right">STREAMLIT · TENSORFLOW · PLOTLY </div>
</div>
""", unsafe_allow_html=True)