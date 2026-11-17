import re
import string

NEGATIONS = {"not", "no", "never", "nor", "cannot", "n't"}


def clean_text(text):
    """Basic cleaning while retaining negation words."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
