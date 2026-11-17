from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "ml" / "sentiment_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python ml/train_model.py"
        )
    return joblib.load(MODEL_PATH)


def predict_sentiment(comment, model=None):
    if model is None:
        model = load_model()

    predicted = model.predict([comment])[0]
    raw_probs = model.predict_proba([comment])[0]
    classes = list(model.classes_)
    probabilities = {label: float(prob) for label, prob in zip(classes, raw_probs)}

    # Ensure all three labels exist in the returned dictionary.
    for label in ("Positive", "Neutral", "Negative"):
        probabilities.setdefault(label, 0.0)

    return predicted, probabilities
