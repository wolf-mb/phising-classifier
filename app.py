import streamlit as st
import joblib
import numpy as np
import re
import os
from urllib.parse import urlparse

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(page_title="PHISHDETECTOR", page_icon="🎣", layout="centered")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center; font-family: 'Arial Black', sans-serif;
        font-size: 4rem; font-weight: 900;
        background: linear-gradient(90deg, #ef4444 0%, #3b82f6 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 5px; margin-top: -20px;
    }
    .sub-text {
        text-align: center; color: #94a3b8;
        font-size: 1.3rem; font-weight: 700; margin-bottom: 30px;
    }
    .result-box-safe {
        background: #f0fdf4; border: 4px solid #22c55e;
        border-radius: 16px; padding: 30px; text-align: center; margin-top: 30px;
    }
    .result-box-danger {
        background: #fef2f2; border: 4px solid #dc2626;
        border-radius: 16px; padding: 30px; text-align: center; margin-top: 30px;
    }
    .result-box-warn {
        background: #fffbeb; border: 4px solid #f59e0b;
        border-radius: 16px; padding: 30px; text-align: center; margin-top: 30px;
    }
    .url-text-display {
        font-family: 'Courier New', Courier, monospace;
        background-color: #1e293b; color: white;
        padding: 12px; border-radius: 8px; font-size: 1.1rem;
        display: inline-block; margin-bottom: 20px;
        word-break: break-all; font-weight: 900; letter-spacing: 1px;
    }
    .confidence-bar-label {
        font-size: 0.95rem; color: #64748b; margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1 class='main-title'>PHISHDETECTOR</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Don't take the bait. Verify if a link is safe before you click.</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("https://img.icons8.com/fluency/512/phishing.png", use_container_width=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "adf.ly", "shorte.st", "rb.gy",
    "cutt.ly", "shorturl.at", "tiny.cc", "x.co"
}

# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None

st.cache_resource.clear()   # ← add this line
model = load_model()

# ── Feature Extraction (12 features matching training_pipeline.py) ────────────
def extract_url_features(url: str) -> np.ndarray:
    """
    Extracts exactly 12 features matching training selected_indices:
    [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 17, 18]
    UCI convention: 1 = phishing signal, -1 = legitimate signal
    Returns shape (1, 12).
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower().split(':')[0]  # strip port if present

    # Index 0: IP address as domain
    f0 = 1 if re.search(r'^\d{1,3}(\.\d{1,3}){3}$', domain) else -1

    # Index 1: URL length > 54
    f1 = 1 if len(url) > 54 else -1

    # Index 2: URL shortener used
    f2 = 1 if any(s in domain for s in SHORTENER_DOMAINS) else -1

    # Index 3: '@' symbol in URL
    f3 = 1 if '@' in url else -1

    # Index 4: '//' after position 7 (redirect trick)
    f4 = 1 if url.rfind('//') > 7 else -1

    # Index 5: Hyphen in domain
    f5 = 1 if '-' in domain else -1

    # Index 6: Subdomain count (more than 2 dots)
    f6 = 1 if domain.count('.') > 2 else -1

    # Index 7: SSL — HTTPS = safe (-1), HTTP = suspicious (1)
    f7 = -1 if parsed.scheme == 'https' else 1

    # Index 10: Non-standard port (not 80 or 443)
    port = parsed.port
    f10 = 1 if (port is not None and port not in (80, 443)) else -1

    # Index 11: "https" token in domain string (e.g. https-paypal-login.com)
    f11 = 1 if 'https' in domain else -1

    # Index 17: Abnormal URL — base domain missing from URL
    try:
        parts = domain.split('.')
        base_domain = '.'.join(parts[-2:]) if len(parts) >= 2 else domain
        f17 = -1 if base_domain in url else 1
    except Exception:
        f17 = 1

    # Index 18: Multiple redirects (more than one '//' beyond scheme)
    redirect_count = url.count('//') - 1
    f18 = 1 if redirect_count > 1 else -1

    return np.array([[f0, f1, f2, f3, f4, f5, f6, f7, f10, f11, f17, f18]])

# ── User Input ────────────────────────────────────────────────────────────────
url_input = st.text_input(
    "URL Input",
    label_visibility="collapsed",
    placeholder="Paste a suspicious link here..."
)
col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
with col_b2:
    scan_button = st.button("Scan URL", use_container_width=True, type="primary")

# ── Inference ─────────────────────────────────────────────────────────────────
if scan_button:
    if not url_input:
        st.toast("⚠️ Please enter a URL to scan.", icon="❌")

    elif model is None:
        st.error("❌ System Error: model.pkl missing. Run the training pipeline first.")

    else:
        with st.spinner("Analyzing domain signature via Random Forest..."):

            parsed_url = urlparse(url_input)
            domain = parsed_url.netloc.lower().split(':')[0]

            # Heuristic override ONLY for raw IP domains — model handles HTTP via Feature 7
            is_ip = bool(re.search(r'^\d{1,3}(\.\d{1,3}){3}$', domain))

            if is_ip:
                phishing_confidence = 0.99
                source_label = "Heuristic (IP-based domain)"
            else:
                ml_features = extract_url_features(url_input)
                probabilities = model.predict_proba(ml_features)[0]
                phishing_confidence = float(probabilities[1])
                source_label = "Random Forest (12-feature)"

        st.markdown("---")

        # ── Result Display ────────────────────────────────────────────────────
        if phishing_confidence > 0.7:
            st.markdown(f"""
            <div class='result-box-danger'>
                <span class='url-text-display'>{url_input}</span>
                <h1 style='color:#dc2626; margin-top:0; font-weight:900; font-size:2.5rem;'>
                    🚨 PHISHING DETECTED
                </h1>
                <h3 style='color:#7f1d1d; margin-bottom:0;'>
                    Do not provide personal information to this site.
                </h3>
            </div>
            """, unsafe_allow_html=True)

        elif phishing_confidence > 0.4:
            st.markdown(f"""
            <div class='result-box-warn'>
                <span class='url-text-display'>{url_input}</span>
                <h1 style='color:#b45309; margin-top:0; font-weight:900; font-size:2.5rem;'>
                    ⚠️ SUSPICIOUS URL
                </h1>
                <h3 style='color:#78350f; margin-bottom:0;'>
                    Proceed with caution. Verify the domain before entering any data.
                </h3>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class='result-box-safe'>
                <span class='url-text-display'>{url_input}</span>
                <h1 style='color:#16a34a; margin-top:0; font-weight:900; font-size:2.5rem;'>
                    ✅ SAFE TO USE
                </h1>
                <h3 style='color:#14532d; margin-bottom:0;'>
                    This domain matches secure baseline patterns.
                </h3>
            </div>
            """, unsafe_allow_html=True)

        # ── Confidence Meter ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(phishing_confidence, text=f"Phishing Confidence: {phishing_confidence*100:.1f}%")
        st.caption(f"Detection source: {source_label}")