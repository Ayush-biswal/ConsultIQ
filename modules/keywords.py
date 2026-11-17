from collections import Counter
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def extract_keywords(comments, limit=12):
    """Return frequent single words and two-word phrases."""
    stop_words = set(ENGLISH_STOP_WORDS)
    unigrams = Counter()
    bigrams = Counter()

    for comment in comments:
        words = re.findall(r"[a-zA-Z][a-zA-Z']+", str(comment).lower())
        useful = [w for w in words if w not in stop_words and len(w) > 2]
        unigrams.update(useful)
        bigrams.update(
            f"{useful[i]} {useful[i+1]}"
            for i in range(len(useful) - 1)
        )

    results = []
    for phrase, count in bigrams.most_common(limit // 2):
        results.append({"keyword": phrase, "count": count, "type": "Phrase"})
    for word, count in unigrams.most_common(limit - len(results)):
        results.append({"keyword": word, "count": count, "type": "Word"})
    return results
