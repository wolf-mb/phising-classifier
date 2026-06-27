import joblib
import numpy as np
import re
import os
from urllib.parse import urlparse

# Known URL shortener domains (Feature Index 2)
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "adf.ly", "shorte.st", "rb.gy",
    "cutt.ly", "shorturl.at", "tiny.cc", "x.co"
}

class PredictionPipeline:
    """
    Handles live URL feature extraction and model inference.
    Extracts exactly 12 features matching training indices:
    [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 17, 18]
    All features follow UCI convention: 1 = phishing, -1 = legitimate
    """

    def __init__(self, model_path="model.pkl"):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        return None

    def extract_features(self, url: str) -> np.ndarray:
        """
        Extracts exactly 12 features in the same order as training.
        Training selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 17, 18]
        Returns shape (1, 12).
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Strip port from domain if present (e.g. "evil.com:8080" -> "evil.com")
        domain = domain.split(':')[0]

        # --- Index 0: IP address used as domain ---
        # Phishing sites often use raw IPs to avoid domain registration traces
        f0 = 1 if re.search(r'^\d{1,3}(\.\d{1,3}){3}$', domain) else -1

        # --- Index 1: URL length > 54 chars ---
        # Legitimate URLs are usually short; phishing URLs pad with subdomains/paths
        f1 = 1 if len(url) > 54 else -1

        # --- Index 2: URL shortener service used ---
        # Phishing links hide behind shorteners to obscure the real domain
        f2 = 1 if any(s in domain for s in SHORTENER_DOMAINS) else -1

        # --- Index 3: '@' symbol in URL ---
        # Browser ignores everything before '@', classic phishing trick
        f3 = 1 if '@' in url else -1

        # --- Index 4: '//' appearing after position 7 ---
        # A double-slash beyond the scheme indicates redirect manipulation
        f4 = 1 if url.rfind('//') > 7 else -1

        # --- Index 5: Hyphen in domain name ---
        # Legitimate domains rarely use hyphens; phishing uses "secure-login-paypal.com"
        f5 = 1 if '-' in domain else -1

        # --- Index 6: Subdomain count (dot count > 2) ---
        # e.g. "login.paypal.secure.evil.com" has 4 dots → phishing
        f6 = 1 if domain.count('.') > 2 else -1

        # --- Index 7: SSL state ---
        # HTTPS = legitimate (-1), HTTP = suspicious (1)
        f7 = -1 if parsed.scheme == 'https' else 1

        # --- Index 10: Non-standard port ---
        # Standard ports: 80 (http), 443 (https). Others are suspicious.
        port = parsed.port
        f10 = 1 if (port is not None and port not in (80, 443)) else -1

        # --- Index 11: "https" token in domain string ---
        # e.g. "https-secure-paypal.com" — phishing trick to appear legitimate
        f11 = 1 if 'https' in domain else -1

        # --- Index 17: Abnormal URL ---
        # Check if the base domain actually appears in the full URL
        # If the hostname is absent from the URL string, something is wrong
        try:
            parts = domain.split('.')
            base_domain = '.'.join(parts[-2:]) if len(parts) >= 2 else domain
            f17 = -1 if base_domain in url else 1
        except Exception:
            f17 = 1

        # --- Index 18: Multiple redirects ---
        # Count '//' occurrences beyond the scheme's '//'
        redirect_count = url.count('//') - 1
        f18 = 1 if redirect_count > 1 else -1

        features = np.array([[f0, f1, f2, f3, f4, f5, f6, f7, f10, f11, f17, f18]])
        return features  # shape: (1, 12)

    def predict_url_safety(self, url: str) -> dict:
        """
        Executes inference workflow.
        Returns dict with: is_phishing (bool), confidence (float), source (str)
        """
        if self.model is None:
            raise FileNotFoundError("model.pkl not found. Run training pipeline first.")

        parsed = urlparse(url)
        domain = parsed.netloc.lower().split(':')[0]

        # Heuristic override ONLY for IP-based domains — near-certain phishing signal
        # HTTP alone is NOT overridden; model handles that via Feature 7
        is_ip = bool(re.search(r'^\d{1,3}(\.\d{1,3}){3}$', domain))
        if is_ip:
            return {
                "is_phishing": True,
                "confidence": 0.99,
                "source": "Heuristic (IP-based domain)"
            }

        # ML inference for everything else
        ml_features = self.extract_features(url)
        probabilities = self.model.predict_proba(ml_features)[0]
        phishing_confidence = float(probabilities[1])

        return {
            "is_phishing": phishing_confidence > 0.5,
            "confidence": phishing_confidence,
            "source": "Random Forest (12-feature)"
        }