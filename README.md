# Spam Email Classifier using Machine Learning

## 📌 Project Overview
This project implements a **Spam Email Classification system** using **Machine Learning** and **Natural Language Processing (NLP)** techniques.  
The system classifies messages as **Spam** or **Ham (Not Spam)** based on patterns learned from a labeled dataset.

The project is **command-line based** and focuses on the complete machine learning pipeline, including data preprocessing, model training, evaluation, and prediction.

---

## 🎯 Objectives
- To automatically detect spam messages
- To apply machine learning techniques to text classification
- To preprocess and clean text data
- To build an efficient spam detection model
- To classify new unseen messages using a trained model

---

## 🧠 Technologies Used
- **Programming Language:** Python
- **Libraries:**
  - Pandas
  - NumPy
  - Scikit-learn
  - NLTK
- **Algorithm:** Multinomial Naive Bayes
- **Feature Extraction:** TF-IDF Vectorizer
- **Dataset:** SMS Spam Collection Dataset

---

## 📂 Project Structure
Spam_Email_Classifier/
│
├── dataset/
│ └── spamm
│
├── model/
│ ├── spam_model.pkl
│ └── vectorizer.pkl
│
├── src/
│ ├── data_preprocessing.py
│ ├── main.py
│ └── predict.py
│
└── README.md

---

## 🗂️ File Description

- **spam.csv** – Dataset used for training the model  
- **data_preprocessing.py** – Cleans and preprocesses text data  
- **main.py** – Trains the model and saves it  
- **predict.py** – Predicts whether a message is spam or not  
- **spam_model.pkl** – Saved trained machine learning model  
- **vectorizer.pkl** – Saved TF-IDF vectorizer  

---

## 🔁 Methodology
1. Load the dataset
2. Preprocess text (lowercase, remove noise)
3. Convert text into numerical features using TF-IDF
4. Train Multinomial Naive Bayes classifier
5. Evaluate model accuracy
6. Save trained model
7. Predict new messages using the saved model

---

## ▶️ How to Run the Project

### 1️⃣ Install Required Libraries
```bash
pip install pandas numpy scikit-learn nltk
📊 Results

Achieved approximately 96–97% accuracy

Successfully classifies spam and non-spam messages

Handles new unseen messages effectively

🧾 Conclusion

The Spam Email Classifier successfully demonstrates the use of machine learning for text classification.
By combining TF-IDF feature extraction with the Naive Bayes algorithm, the system efficiently identifies spam messages and improves message filtering accuracy.

🚀 Future Enhancements

Add a graphical user interface (GUI)

Deploy as a web application

Use deep learning models for improved accuracy

Integrate with real-time email systems

👨‍💻 Developed By

Piush Saha
