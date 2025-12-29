import pickle
from data_preprocessing import preprocess_text

# Load trained model and vectorizer
model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

print("Spam Email Classifier (type 'exit' to stop)\n")

while True:
    message = input("Congratulations! You won a free iPhone\nHey, are we meeting tomorrow? ")

    if message.lower() == "exit":
        break

    clean = preprocess_text(message)
    vector = vectorizer.transform([clean])

    prediction = model.predict(vector)[0]

    if prediction == 1:
        print("Result: 🚨 Spam\n")
    else:
        print("Result: ✅ Not Spam\n")
