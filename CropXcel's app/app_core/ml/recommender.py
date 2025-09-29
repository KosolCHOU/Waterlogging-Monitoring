# app_core/ml/recommender.py
import os
import threading
import numpy as np
from django.conf import settings

# If you used joblib for scikit-learn:
try:
    import joblib
except Exception:
    joblib = None

_MODEL = None
_LOCK = threading.Lock()

def _model_path() -> str:
    # e.g. <BASE_DIR>/app_core/ml/crop_model.pkl
    return os.path.join(settings.BASE_DIR, "app_core", "ml", "crop_model.pkl")

def get_model():
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                path = _model_path()
                if (joblib is None) or (not os.path.exists(path)):
                    raise RuntimeError(f"Model not available at {path}")
                _MODEL = joblib.load(path)
    return _MODEL

def predict_crop(features: dict):
    """
    features keys (float): N, P, K, temperature, humidity, pH, rainfall
    Returns: (label, prob_dict)  # prob_dict may be {} if model lacks predict_proba
    """
    order = ["N", "P", "K", "temperature", "humidity", "pH", "rainfall"]
    X = np.array([[float(features[k]) for k in order]], dtype=float)

    model = get_model()

    # Basic: label
    label = model.predict(X)[0]

    # Optional: probabilities if supported
    prob_map = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        # If classifier has classes_
        classes = getattr(model, "classes_", [str(i) for i in range(len(probs))])
        prob_map = {str(c): float(p) for c, p in zip(classes, probs)}

    return str(label), prob_map
