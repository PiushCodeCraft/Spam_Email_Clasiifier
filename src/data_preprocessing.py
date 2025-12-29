import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove urls, numbers & punctuation only
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    words = text.split()

    filtered_words = []

    for word in words:
        # keep at least some short words, only remove very common ones
        if word not in stop_words:
            filtered_words.append(ps.stem(word))

    cleaned = " ".join(filtered_words).strip()

    # IMPORTANT — if empty, return original lowercase text (fallback)
    return cleaned if cleaned != "" else text.strip()
