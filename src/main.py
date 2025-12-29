import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from data_preprocessing import preprocess_text

# ---------------------------
# 1️⃣ Load dataset
# ---------------------------
data = pd.read_csv("dataset/spamm", sep="\t", header=None, names=["label", "message"])

# keep only valid messages
data = data.dropna(subset=["message"])

# ---------------------------
# 2️⃣ Preprocess text
# ---------------------------
data["clean_text"] = data["message"].apply(preprocess_text)

# remove rows where preprocessing returned None / NaN / empty
data = data[data["clean_text"].notna()]
data["clean_text"] = data["clean_text"].astype(str)
data = data[data["clean_text"].str.strip() != ""]

# ---------------------------
# 3️⃣ Features and labels
# ---------------------------
X = data["clean_text"]

print("Total rows:", len(X))
print(X.head(20))
print(X.unique()[:10])

y = data["label"].map({"ham": 0, "spam": 1})

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# ---------------------------
# 4️⃣ Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 5️⃣ Train model
# ---------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# ---------------------------
# 6️⃣ Evaluate
# ---------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---------------------------
# 7️⃣ Save model + vectorizer
# ---------------------------
os.makedirs("model", exist_ok=True)

with open("model/spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model saved successfully!")







# import pandas as pd

# data = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# print(data.head())
# print(data.columns)
