RISK_KEYWORDS = {
    "unsafe", "dangerous", "hazard", "broken", "injury", "accident",
    "urgent", "emergency", "not working", "exposed wire", "fire",
    "collapsed", "pothole", "flooding"
}


def detect_priority(comment, sentiment, probabilities):
    """
    Simple review flag:
    - Any risk phrase can flag a comment, regardless of sentiment.
    - A negative prediction with probability >= 0.80 is also flagged.
    This is an aid for human review, not emergency detection.
    """
    text = str(comment).lower()
    matched = sorted(term for term in RISK_KEYWORDS if term in text)
    reasons = []

    if matched:
        reasons.append("Risk term(s): " + ", ".join(matched))

    negative_probability = float(probabilities.get("Negative", 0))
    if sentiment == "Negative" and negative_probability >= 0.80:
        reasons.append(f"High negative sentiment probability ({negative_probability:.0%})")

    return bool(reasons), "; ".join(reasons)
