"""
Aircraft Damage Classification & Captioning
Final Project – Streamlit UI
"""

import os, sys, warnings, io, random, time

# ─── Add local pip packages to path (transformers, torch, etc.) ───────────────
_pip_pkg_dir = r"C:\Users\Orion\pip_packages"
if os.path.isdir(_pip_pkg_dir) and _pip_pkg_dir not in sys.path:
    sys.path.insert(0, _pip_pkg_dir)
warnings.filterwarnings("ignore")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AeroInspect AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0d1a 0%, #0e1324 50%, #111827 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
    border-right: 1px solid #1e2d3d;
}

[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Hero Header ── */
.hero-header {
    background: linear-gradient(135deg, #1e3a5f, #0f2342, #1a1040);
    border: 1px solid #2563eb30;
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, #2563eb15 0%, transparent 70%);
    animation: pulse 4s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.05); opacity: 1; }
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 0.5rem;
    font-weight: 400;
}

/* ── Cards ── */
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(37, 99, 235, 0.2);
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    color: #64748b;
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* ── Section headings ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #1e3a5f;
}

.section-header h2 {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0;
}

/* ── Task badge ── */
.task-badge {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: inline-block;
    margin-bottom: 0.5rem;
}

/* ── Result boxes ── */
.result-box {
    background: linear-gradient(135deg, #0f2342, #1a1040);
    border: 1px solid #2563eb50;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    color: #a5f3fc;
}

.caption-result {
    background: linear-gradient(135deg, #0f2342 0%, #1a0f42 100%);
    border: 1px solid #7c3aed50;
    border-left: 4px solid #7c3aed;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: #e2e8f0;
    font-size: 1.05rem;
    line-height: 1.6;
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    font-family: 'Inter', sans-serif !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(37,99,235,0.4) !important;
}

div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #2563eb, #7c3aed, #34d399) !important;
}

.stSelectbox > div, .stSlider > div {
    color: #e2e8f0 !important;
}

.stAlert {
    border-radius: 12px !important;
}

/* ── Tables ── */
table {
    background: #1e293b !important;
    border-radius: 8px;
}

th { background: #0f172a !important; color: #60a5fa !important; }
td { color: #e2e8f0 !important; }

/* ── Image Display ── */
.stImage > img {
    border-radius: 12px;
    border: 1px solid #334155;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #60a5fa !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #1e293b !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Info boxes ── */
.info-box {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #94a3b8;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* ── Step indicators ── */
.step-list {
    list-style: none;
    padding: 0;
    margin: 0;
}
.step-list li {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    padding: 0.4rem 0;
    color: #94a3b8;
    font-size: 0.95rem;
}
.step-num {
    background: #2563eb;
    color: white;
    border-radius: 50%;
    width: 22px;
    height: 22px;
    min-width: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "dataset_ready": False,
        "model_trained": False,
        "train_history": None,
        "test_loss": None,
        "test_accuracy": None,
        "model": None,
        "generators": None,
        "blip_loaded": False,
        "n_epochs": 5,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:3rem;'>✈️</div>
        <div style='font-size:1.2rem; font-weight:700; color:#60a5fa;'>AeroInspect AI</div>
        <div style='font-size:0.8rem; color:#64748b; margin-top:0.3rem;'>Aircraft Damage Analysis</div>
    </div>
    <hr style='border-color:#1e2d3d; margin: 1rem 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📦 Dataset Setup",
            "🧠 Train Classifier",
            "📊 Training Results",
            "🔍 Evaluate & Predict",
            "💬 Image Captioning",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#1e2d3d; margin: 1rem 0;'>", unsafe_allow_html=True)

    # Status panel
    st.markdown("**System Status**")
    ds_icon = "✅" if st.session_state.dataset_ready else "⏳"
    tr_icon = "✅" if st.session_state.model_trained else "⏳"

    st.markdown(f"""
    <div style='font-size:0.85rem; color:#94a3b8; line-height:2;'>
        {ds_icon} Dataset<br>
        {tr_icon} Model Trained<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2d3d; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#475569; text-align:center;'>
        Deep Learning Final Project<br>
        VGG16 + BLIP Transformers
    </div>
    """, unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────────────────────
def hero():
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">✈️ AeroInspect AI</div>
        <div class="hero-subtitle">
            Aircraft Damage Classification &amp; Captioning — Deep Learning Final Project
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── PAGE: Overview ────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    hero()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">VGG16</div>
            <div class="metric-label">Base Model</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">BLIP</div>
            <div class="metric-label">Caption Model</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">10</div>
            <div class="metric-label">Tasks Covered</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="section-header"><h2>🎯 Project Overview</h2></div>
        <div class="info-box">
        Aircraft damage detection is essential for maintaining safety and longevity.
        This project automates the classification of aircraft surface damage into two categories:
        <strong style='color:#60a5fa;'>Dent</strong> and <strong style='color:#f472b6;'>Crack</strong>.
        <br><br>
        We leverage <strong>feature extraction with VGG16</strong> (pre-trained on ImageNet) 
        and a custom classifier head, then use <strong>BLIP</strong> (Bootstrapping Language-Image Pretraining)
        to generate natural language captions and summaries of the damage.
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="section-header"><h2>📋 Task Checklist</h2></div>
        <div class="info-box">
        <ul class="step-list">
            <li><span class="step-num">1</span> Create Validation Generator</li>
            <li><span class="step-num">2</span> Create Test Generator</li>
            <li><span class="step-num">3</span> Load Pre-trained VGG16</li>
            <li><span class="step-num">4</span> Compile the Model</li>
            <li><span class="step-num">5</span> Train the Model</li>
            <li><span class="step-num">6</span> Plot Accuracy Curves</li>
            <li><span class="step-num">7</span> Visualize Predictions</li>
            <li><span class="step-num">8</span> Implement BLIP Helper Function</li>
            <li><span class="step-num">9</span> Generate Image Caption</li>
            <li><span class="step-num">10</span> Generate Image Summary</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header"><h2>🛣️ How to Use This App</h2></div>
    <div class="info-box">
    <ol>
        <li>Go to <strong>📦 Dataset Setup</strong> → Download & explore the aircraft dataset.</li>
        <li>Go to <strong>🧠 Train Classifier</strong> → Train the VGG16 model (Tasks 1–5).</li>
        <li>Go to <strong>📊 Training Results</strong> → View accuracy/loss plots (Task 6).</li>
        <li>Go to <strong>🔍 Evaluate & Predict</strong> → See predictions on test images (Task 7).</li>
        <li>Go to <strong>💬 Image Captioning</strong> → Generate captions & summaries with BLIP (Tasks 8–10).</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

# ─── PAGE: Dataset Setup ───────────────────────────────────────────────────────
elif page == "📦 Dataset Setup":
    hero()
    st.markdown('<div class="section-header"><h2>📦 Dataset Setup</h2></div>', unsafe_allow_html=True)

    from download_dataset import dataset_exists, download_and_extract, get_dataset_path, count_images

    st.markdown("""
    <div class="info-box">
    The <strong>Aircraft Damage Dataset</strong> contains images of aircraft surfaces labelled as 
    <strong style='color:#60a5fa;'>dent</strong> or <strong style='color:#f472b6;'>crack</strong>.
    The dataset is split into <code>train</code>, <code>valid</code>, and <code>test</code> directories.
    <br><br>
    Source: <a href="https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk" target="_blank" style='color:#60a5fa;'>Roboflow Aircraft Dataset</a> (License: CC BY 4)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if dataset_exists():
        st.session_state.dataset_ready = True
        st.success("✅ Dataset already downloaded and ready!")
    
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        if st.button("⬇️ Download Dataset", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(msg):
                status_text.markdown(f'<div class="result-box">{msg}</div>', unsafe_allow_html=True)

            with st.spinner("Downloading aircraft dataset (~300 MB)..."):
                try:
                    update_progress("🌐 Connecting to server...")
                    progress_bar.progress(10)
                    download_and_extract(progress_callback=update_progress)
                    progress_bar.progress(100)
                    st.session_state.dataset_ready = True
                    st.success("✅ Dataset downloaded and extracted successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Download failed: {e}")

    with col_info:
        st.markdown("""
        <div class="info-box">
        <strong>📁 Expected folder structure:</strong><br><br>
        <code>aircraft_damage_dataset_v1/</code><br>
        &nbsp;&nbsp;├── <code>train/</code> &nbsp;&nbsp; dent/ &nbsp; crack/<br>
        &nbsp;&nbsp;├── <code>valid/</code> &nbsp;&nbsp; dent/ &nbsp; crack/<br>
        &nbsp;&nbsp;└── <code>test/</code> &nbsp;&nbsp;&nbsp; dent/ &nbsp; crack/
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.dataset_ready:
        dataset_path = get_dataset_path()
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header"><h2>📊 Dataset Statistics</h2></div>', unsafe_allow_html=True)

        splits = ["train", "valid", "test"]
        cols = st.columns(3)
        total_images = 0

        for i, split in enumerate(splits):
            split_dir = os.path.join(dataset_path, split)
            counts = count_images(split_dir)
            total = sum(counts.values())
            total_images += total
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total}</div>
                    <div class="metric-label">{split.capitalize()} Images</div>
                    <div style='margin-top:0.7rem; font-size:0.85rem; color:#64748b;'>
                        {"  |  ".join([f"{k}: {v}" for k, v in counts.items()])}
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header"><h2>🖼️ Sample Images</h2></div>', unsafe_allow_html=True)

        for split in ["train"]:
            split_dir = os.path.join(dataset_path, split)
            sample_cols = st.columns(4)
            all_images = []
            for cls in ["dent", "crack"]:
                cls_dir = os.path.join(split_dir, cls)
                if os.path.exists(cls_dir):
                    imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    for fname in random.sample(imgs, min(2, len(imgs))):
                        all_images.append((os.path.join(cls_dir, fname), cls))

            for idx, (img_path, cls) in enumerate(all_images[:4]):
                with sample_cols[idx]:
                    img = Image.open(img_path)
                    badge_color = "#60a5fa" if cls == "dent" else "#f472b6"
                    st.image(img, use_container_width=True)
                    st.markdown(f"""
                    <div style='text-align:center; margin-top:0.3rem;'>
                        <span style='background:{badge_color}22; color:{badge_color}; 
                        border:1px solid {badge_color}50; border-radius:999px; 
                        padding:0.2rem 0.8rem; font-size:0.8rem; font-weight:600;'>
                        {cls.upper()}
                        </span>
                    </div>""", unsafe_allow_html=True)

# ─── PAGE: Train Classifier ────────────────────────────────────────────────────
elif page == "🧠 Train Classifier":
    hero()
    st.markdown('<div class="section-header"><h2>🧠 VGG16 Classifier Training</h2></div>', unsafe_allow_html=True)

    if not st.session_state.dataset_ready:
        st.warning("⚠️ Please download the dataset first (📦 Dataset Setup page).")
    else:
        col_cfg, col_arch = st.columns([1, 2])

        with col_cfg:
            st.markdown("**⚙️ Training Configuration**")
            n_epochs = st.slider("Epochs", min_value=1, max_value=20, value=5, key="epoch_slider")
            st.session_state.n_epochs = n_epochs

            st.markdown("""
            <div class="info-box" style='margin-top:1rem;'>
            <strong>Fixed parameters:</strong><br>
            • Batch size: <code>32</code><br>
            • Image size: <code>224 × 224</code><br>
            • Optimizer: <code>Adam (lr=0.0001)</code><br>
            • Loss: <code>Binary Crossentropy</code>
            </div>""", unsafe_allow_html=True)

        with col_arch:
            st.markdown("**🏗️ Model Architecture**")
            st.markdown("""
            <div class="info-box">
            <strong>Task 3 – VGG16 Base:</strong><br>
            &nbsp;&nbsp;→ Pre-trained on ImageNet (weights frozen)<br>
            &nbsp;&nbsp;→ <code>include_top=False</code> → Flatten<br><br>
            <strong>Custom Head (Task 4 – Compile):</strong><br>
            &nbsp;&nbsp;→ Dense(512, ReLU) → Dropout(0.3)<br>
            &nbsp;&nbsp;→ Dense(512, ReLU) → Dropout(0.3)<br>
            &nbsp;&nbsp;→ Dense(1, Sigmoid) — Binary Output<br><br>
            <strong>Tasks 1 & 2 – Data Generators:</strong><br>
            &nbsp;&nbsp;→ ImageDataGenerator (rescale 1/255)<br>
            &nbsp;&nbsp;→ Valid & Test: shuffle=False, binary mode
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Start Training (Tasks 1–5)", use_container_width=True):
            import train_model as tm
            from download_dataset import get_dataset_path

            dataset_path = get_dataset_path()
            train_dir = os.path.join(dataset_path, "train")
            valid_dir = os.path.join(dataset_path, "valid")
            test_dir  = os.path.join(dataset_path, "test")

            progress_container = st.container()
            log_area = st.empty()

            with st.spinner("🔄 Building data generators (Tasks 1 & 2)..."):
                try:
                    train_gen, valid_gen, test_gen = tm.build_generators(train_dir, valid_dir, test_dir)
                    st.session_state.generators = (train_gen, valid_gen, test_gen)
                    st.success(f"✅ Task 1 & 2 complete — generators ready. Train: {train_gen.samples} | Valid: {valid_gen.samples} | Test: {test_gen.samples}")
                except Exception as e:
                    st.error(f"❌ Generator error: {e}")
                    st.stop()

            with st.spinner("🔄 Loading VGG16 (Task 3) & compiling (Task 4)..."):
                try:
                    model = tm.build_model()
                    st.success("✅ Task 3 & 4 complete — VGG16 loaded & model compiled.")
                    with st.expander("📋 Model Summary"):
                        summary_buf = io.StringIO()
                        model.summary(print_fn=lambda x: summary_buf.write(x + "\n"))
                        st.code(summary_buf.getvalue(), language="text")
                except Exception as e:
                    st.error(f"❌ Model build error: {e}")
                    st.stop()

            st.markdown("**🏋️ Task 5 – Training the Model...**")
            epoch_progress = st.progress(0)
            epoch_status = st.empty()
            
            history_data = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
            
            chart_col1, chart_col2 = st.columns(2)
            loss_chart = chart_col1.empty()
            acc_chart = chart_col2.empty()
            
            # Custom callback for live Streamlit updates
            try:
                import keras

                class StreamlitCallback(keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}
                        history_data["loss"].append(logs.get("loss", 0))
                        history_data["val_loss"].append(logs.get("val_loss", 0))
                        history_data["accuracy"].append(logs.get("accuracy", 0))
                        history_data["val_accuracy"].append(logs.get("val_accuracy", 0))

                        pct = int((epoch + 1) / n_epochs * 100)
                        epoch_progress.progress(pct)
                        epoch_status.markdown(
                            f'<div class="result-box">'
                            f'Epoch {epoch+1}/{n_epochs} — '
                            f'Loss: {logs.get("loss",0):.4f} | '
                            f'Acc: {logs.get("accuracy",0):.4f} | '
                            f'Val Loss: {logs.get("val_loss",0):.4f} | '
                            f'Val Acc: {logs.get("val_accuracy",0):.4f}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        # Live loss chart
                        fig_l, ax_l = plt.subplots(figsize=(5, 3))
                        fig_l.patch.set_facecolor('#0e1117')
                        ax_l.set_facecolor('#1a1d2e')
                        ax_l.plot(history_data["loss"], color="#89b4fa", label="Train")
                        ax_l.plot(history_data["val_loss"], color="#fab387", label="Val")
                        ax_l.set_title("Loss", color="#cdd6f4")
                        ax_l.tick_params(colors="#cdd6f4")
                        ax_l.legend(facecolor="#313244", labelcolor="white")
                        ax_l.spines[:].set_color("#313244")
                        loss_chart.pyplot(fig_l)
                        plt.close(fig_l)

                        # Live acc chart
                        fig_a, ax_a = plt.subplots(figsize=(5, 3))
                        fig_a.patch.set_facecolor('#0e1117')
                        ax_a.set_facecolor('#1a1d2e')
                        ax_a.plot(history_data["accuracy"], color="#a6e3a1", label="Train")
                        ax_a.plot(history_data["val_accuracy"], color="#f38ba8", label="Val")
                        ax_a.set_title("Accuracy", color="#cdd6f4")
                        ax_a.tick_params(colors="#cdd6f4")
                        ax_a.legend(facecolor="#313244", labelcolor="white")
                        ax_a.spines[:].set_color("#313244")
                        acc_chart.pyplot(fig_a)
                        plt.close(fig_a)

                history = model.fit(
                    train_gen,
                    epochs=n_epochs,
                    validation_data=valid_gen,
                    callbacks=[StreamlitCallback()],
                )

                st.session_state.model = model
                st.session_state.train_history = history.history
                st.session_state.model_trained = True
                
                # Save model
                model_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "aircraft_vgg16_model.keras"
                )
                model.save(model_path)

                st.success("🎉 Task 5 Complete! Model trained and saved successfully!")
                st.balloons()

            except Exception as e:
                st.error(f"❌ Training error: {e}")
                import traceback
                st.code(traceback.format_exc())

# ─── PAGE: Training Results ────────────────────────────────────────────────────
elif page == "📊 Training Results":
    hero()
    st.markdown('<div class="section-header"><h2>📊 Training Results</h2></div>', unsafe_allow_html=True)

    if not st.session_state.model_trained or st.session_state.train_history is None:
        st.warning("⚠️ Please train the model first (🧠 Train Classifier page).")
    else:
        import train_model as tm

        history = st.session_state.train_history
        n_ep = len(history["loss"])

        # Metrics summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{history['accuracy'][-1]:.3f}</div>
                <div class="metric-label">Final Train Acc</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{history['val_accuracy'][-1]:.3f}</div>
                <div class="metric-label">Final Val Acc</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{history['loss'][-1]:.4f}</div>
                <div class="metric-label">Final Train Loss</div></div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{n_ep}</div>
                <div class="metric-label">Epochs Trained</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Loss curves
        st.markdown('<div class="section-header"><h2>📉 Loss Curves</h2></div>', unsafe_allow_html=True)
        fig_loss = tm.plot_loss_curves(history)
        st.pyplot(fig_loss)
        plt.close()

        st.markdown("<br>", unsafe_allow_html=True)

        # Task 6 – Accuracy curves
        st.markdown("""
        <span class="task-badge">Task 6</span>
        <div class="section-header"><h2>📈 Accuracy Curves</h2></div>""", unsafe_allow_html=True)
        fig_acc = tm.plot_accuracy_curves(history)
        st.pyplot(fig_acc)
        plt.close()

        st.markdown("<br>", unsafe_allow_html=True)

        # Full history table
        with st.expander("📋 Full Training History Table"):
            import pandas as pd
            df = pd.DataFrame(history)
            df.index = range(1, len(df) + 1)
            df.index.name = "Epoch"
            df.columns = ["Train Loss", "Train Acc", "Val Loss", "Val Acc"]
            st.dataframe(df.style.format("{:.4f}"), use_container_width=True)

# ─── PAGE: Evaluate & Predict ─────────────────────────────────────────────────
elif page == "🔍 Evaluate & Predict":
    hero()
    st.markdown('<div class="section-header"><h2>🔍 Model Evaluation & Predictions</h2></div>', unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first (🧠 Train Classifier page).")
    else:
        import train_model as tm

        model = st.session_state.model
        _, _, test_gen = st.session_state.generators

        # Evaluate
        col_eval, col_pred = st.columns([1, 1])

        with col_eval:
            st.markdown("**📏 Model Evaluation on Test Set**")
            if st.button("🔬 Evaluate Model", use_container_width=True):
                with st.spinner("Evaluating..."):
                    try:
                        test_gen.reset()
                        test_loss, test_acc = tm.evaluate_model(model, test_gen)
                        st.session_state.test_loss = test_loss
                        st.session_state.test_accuracy = test_acc
                    except Exception as e:
                        st.error(f"Evaluation error: {e}")

            if st.session_state.test_loss is not None:
                st.markdown(f"""
                <div class="metric-card" style='margin-top:1rem;'>
                    <div class="metric-value">{st.session_state.test_accuracy:.4f}</div>
                    <div class="metric-label">Test Accuracy</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-card" style='margin-top:0.5rem;'>
                    <div class="metric-value">{st.session_state.test_loss:.4f}</div>
                    <div class="metric-label">Test Loss</div>
                </div>""", unsafe_allow_html=True)

        with col_pred:
            st.markdown("**🖼️ Upload Your Own Image**")
            uploaded = st.file_uploader(
                "Upload an aircraft image (JPG/PNG)",
                type=["jpg", "jpeg", "png"],
                key="eval_upload"
            )
            if uploaded:
                pil_img = Image.open(uploaded).convert("RGB")
                st.image(pil_img, caption="Uploaded Image", use_container_width=True)

                if st.button("🔮 Classify This Image", use_container_width=True):
                    with st.spinner("Classifying..."):
                        import numpy as np
                        img_resized = pil_img.resize((224, 224))
                        img_array = np.array(img_resized) / 255.0
                        img_array = np.expand_dims(img_array, 0)
                        pred = model.predict(img_array)[0][0]
                        class_label = "crack" if pred > 0.5 else "dent"
                        confidence = pred if pred > 0.5 else 1 - pred

                    badge_color = "#f472b6" if class_label == "crack" else "#60a5fa"
                    st.markdown(f"""
                    <div style='text-align:center; margin-top:1rem;'>
                        <div style='font-size:1rem; color:#94a3b8; margin-bottom:0.5rem;'>Prediction</div>
                        <span style='background:{badge_color}22; color:{badge_color}; 
                        border:1px solid {badge_color}50; border-radius:999px; 
                        padding:0.5rem 1.5rem; font-size:1.4rem; font-weight:700;'>
                        {class_label.upper()}
                        </span>
                        <div style='margin-top:0.8rem; color:#94a3b8; font-size:0.9rem;'>
                        Confidence: <strong style='color:#e2e8f0;'>{confidence:.1%}</strong>
                        </div>
                    </div>""", unsafe_allow_html=True)

        # Task 7 – Prediction Grid
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <span class="task-badge">Task 7</span>
        <div class="section-header"><h2>🖼️ Visualizing Test Predictions</h2></div>""", unsafe_allow_html=True)

        num_grid = st.slider("Number of test images to display", 3, 12, 9, 3, key="grid_slider")

        if st.button("🎲 Show Prediction Grid", use_container_width=True):
            with st.spinner("Generating predictions..."):
                try:
                    test_gen.reset()
                    fig_grid = tm.plot_prediction_grid(test_gen, model, num_images=num_grid)
                    st.pyplot(fig_grid)
                    plt.close()
                    st.markdown("""
                    <div class="info-box" style='margin-top:1rem;'>
                    <strong style='color:#a6e3a1;'>Green title</strong> = Correct prediction &nbsp;|&nbsp; 
                    <strong style='color:#f38ba8;'>Red title</strong> = Incorrect prediction<br>
                    <em>Note: Due to the inherent nature of neural networks, predictions may vary.</em>
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Visualization error: {e}")

# ─── PAGE: Image Captioning ────────────────────────────────────────────────────
elif page == "💬 Image Captioning":
    hero()
    st.markdown('<div class="section-header"><h2>💬 BLIP Image Captioning & Summarization</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>BLIP</strong> (Bootstrapping Language-Image Pretraining) is a state-of-the-art vision-language model 
    that generates human-readable descriptions of images. Here we use <code>Salesforce/blip-image-captioning-base</code>
    from 🤗 Hugging Face.<br><br>
    This page covers <strong>Tasks 8, 9, and 10</strong>.
    </div>
    <br>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📁 Dataset Images (Tasks 9 & 10)", "📤 Upload Custom Image"])

    with tab1:
        st.markdown("""
        <span class="task-badge">Tasks 9 & 10</span>
        """, unsafe_allow_html=True)

        from download_dataset import dataset_exists, get_dataset_path
        import caption_model as cm

        if not dataset_exists():
            st.warning("⚠️ Please download the dataset first (📦 Dataset Setup page).")
        else:
            dataset_path = get_dataset_path()
            test_dent_dir = os.path.join(dataset_path, "test", "dent")
            test_crack_dir = os.path.join(dataset_path, "test", "crack")

            all_test_imgs = []
            for cls, d in [("dent", test_dent_dir), ("crack", test_crack_dir)]:
                if os.path.exists(d):
                    for f in os.listdir(d):
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                            all_test_imgs.append((os.path.join(d, f), cls))

            if all_test_imgs:
                col_sel, col_img = st.columns([1, 1])

                with col_sel:
                    # Mirror the example from the project notebook
                    default_dent = "149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg"
                    default_path = os.path.join(test_dent_dir, default_dent)
                    
                    options = [os.path.basename(p) for p, _ in all_test_imgs[:20]]
                    if default_dent in options:
                        default_idx = options.index(default_dent)
                    else:
                        default_idx = 0

                    selected_name = st.selectbox("Select a test image", options, index=default_idx)
                    selected_full = next(p for p, _ in all_test_imgs if os.path.basename(p) == selected_name)
                    selected_cls  = next(c for p, c in all_test_imgs if os.path.basename(p) == selected_name)

                with col_img:
                    img = Image.open(selected_full)
                    badge_color = "#60a5fa" if selected_cls == "dent" else "#f472b6"
                    st.image(img, use_container_width=True)
                    st.markdown(f"""
                    <div style='text-align:center;'>
                    <span style='background:{badge_color}22; color:{badge_color}; 
                    border:1px solid {badge_color}50; border-radius:999px; 
                    padding:0.2rem 0.8rem; font-size:0.85rem; font-weight:600;'>
                    {selected_cls.upper()}</span></div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                if st.button("🤖 Generate Caption & Summary (Tasks 9 & 10)", use_container_width=True):
                    with st.spinner("Loading BLIP model (first time may take a minute)..."):
                        try:
                            caption, summary = cm.generate_caption_and_summary(selected_full)

                            st.markdown("**📝 Task 9 – Generated Caption:**")
                            st.markdown(f'<div class="caption-result">🖼️ {caption}</div>', unsafe_allow_html=True)
                            
                            st.markdown("<br>**📄 Task 10 – Generated Summary:**")
                            st.markdown(f'<div class="caption-result">📋 {summary}</div>', unsafe_allow_html=True)

                            # Show the code used
                            with st.expander("📋 Task 8 – Helper Function Code"):
                                st.code("""
# Task 8: Helper function using BlipCaptionSummaryLayer
def generate_text(image_path, task):
    processor, model = load_blip_model()
    blip_layer = BlipCaptionSummaryLayer(processor, model)
    return blip_layer(image_path, task)

# Task 9: Generate Caption
image_path = tf.constant("path/to/image.jpg")
caption = generate_text(image_path, tf.constant("caption"))
print("Caption:", caption.numpy().decode("utf-8"))

# Task 10: Generate Summary
summary = generate_text(image_path, tf.constant("summary"))
print("Summary:", summary.numpy().decode("utf-8"))
                                """, language="python")

                        except Exception as e:
                            st.error(f"❌ BLIP error: {e}")
                            import traceback
                            st.code(traceback.format_exc())

    with tab2:
        st.markdown("**📤 Upload your own aircraft image for captioning**")
        uploaded_caption = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            key="caption_upload"
        )
        if uploaded_caption:
            pil_img = Image.open(uploaded_caption).convert("RGB")
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                st.image(pil_img, caption="Uploaded Image", use_container_width=True)
            with col_u2:
                if st.button("✨ Generate Caption + Summary", use_container_width=True):
                    import caption_model as cm
                    with st.spinner("Running BLIP model..."):
                        try:
                            caption = cm.generate_from_pil_image(pil_img, "caption")
                            summary = cm.generate_from_pil_image(pil_img, "summary")
                            st.markdown("**📝 Caption:**")
                            st.markdown(f'<div class="caption-result">🖼️ {caption}</div>', unsafe_allow_html=True)
                            st.markdown("<br>**📄 Summary:**")
                            st.markdown(f'<div class="caption-result">📋 {summary}</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error: {e}")
