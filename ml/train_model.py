
from pathlib import Path
import json
import joblib

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "tweeteval" / "sentiment"

MODEL_PATH = ROOT / "ml" / "sentiment_model.joblib"

METRICS_PATH = ROOT / "ml" / "metrics.json"

MATRIX_PATH = ROOT / "ml" / "confusion_matrix.png"


# ==========================================
# 2. SENTIMENT LABELS
# ==========================================

LABELS = {
    "0": "Negative",
    "1": "Neutral",
    "2": "Positive"
}

CLASS_NAMES = ["Negative", "Neutral", "Positive"]


# ==========================================
# 3. LOAD DATASET
# ==========================================

def load_dataset(split):

    text_file = DATA / f"{split}_text.txt"
    label_file = DATA / f"{split}_labels.txt"

    if not text_file.exists() or not label_file.exists():
        raise FileNotFoundError(
            f"Missing {split} dataset files. "
            "Check data/tweeteval/sentiment/"
        )

    texts = text_file.read_text(
        encoding="utf-8"
    ).splitlines()

    labels = label_file.read_text(
        encoding="utf-8"
    ).splitlines()

    if len(texts) != len(labels):
        raise ValueError(
            f"{split}: Text and label counts do not match."
        )

    sentiments = [
        LABELS[label.strip()]
        for label in labels
    ]

    return texts, sentiments


# ==========================================
# 4. MAIN TRAINING FUNCTION
# ==========================================

def main():

    print("\n================================")
    print("       CONSULTIQ ML TRAINING")
    print("================================")

    print("\nLoading TweetEval dataset...")

    X_train, y_train = load_dataset("train")

    X_val, y_val = load_dataset("val")

    X_test, y_test = load_dataset("test")

    print("\nDataset Information")

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Testing samples:", len(X_test))

    # ======================================
    # 5. BUILD ML PIPELINE
    # ======================================

    print("\nBuilding TF-IDF + Logistic Regression...")

    model = Pipeline([

        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=50000,
            min_df=2,
            sublinear_tf=True
        )),

        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))

    ])

    # ======================================
    # 6. TRAIN MODEL
    # ======================================

    print("\nTraining model...")

    model.fit(X_train, y_train)

    print("\nModel training completed!")

    # ======================================
    # 7. VALIDATION
    # ======================================

    print("\nEvaluating validation dataset...")

    val_predictions = model.predict(X_val)

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    print(
        "Validation Accuracy:",
        round(val_accuracy, 4)
    )

    # ======================================
    # 8. TESTING
    # ======================================

    print("\nEvaluating independent test dataset...")

    test_predictions = model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    report = classification_report(
        y_test,
        test_predictions,
        labels=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    print("\nTEST ACCURACY:", round(test_accuracy, 4))

    print("\nCLASSIFICATION REPORT")

    print(classification_report(
        y_test,
        test_predictions,
        labels=CLASS_NAMES,
        zero_division=0
    ))

    # ======================================
    # 9. SAVE TRAINED MODEL
    # ======================================

    joblib.dump(model, MODEL_PATH)

    print("\nModel saved successfully!")
    print(MODEL_PATH)

    # ======================================
    # 10. SAVE EVALUATION METRICS
    # ======================================

    metrics = {
        "validation_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "testing_samples": len(X_test),
        "classification_report": report
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(metrics, file, indent=4)

    print("\nMetrics saved:", METRICS_PATH)

    # ======================================
    # 11. CONFUSION MATRIX
    # ======================================

    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=CLASS_NAMES
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    plt.title("ConsultIQ - Sentiment Confusion Matrix")

    plt.tight_layout()

    plt.savefig(MATRIX_PATH, dpi=300)

    plt.close()

    print("\nConfusion matrix saved:", MATRIX_PATH)

    # ======================================
    # 12. COMPLETION
    # ======================================

    print("\n================================")
    print("   TRAINING COMPLETED!")
    print("================================")


if __name__ == "__main__":
    main()
