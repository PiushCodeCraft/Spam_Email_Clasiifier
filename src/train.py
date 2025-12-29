# """
# Model Training Module
# Trains the spam classifier using Multinomial Naive Bayes
# """

# import pandas as pd
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import sys
# import os

# # Add parent directory to path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src.preprocess import TextPreprocessor


# class SpamClassifierTrainer:
#     """Class for training spam classification model"""
    
#     def __init__(self, dataset_path='dataset/spam.csv'):
#         self.dataset_path = dataset_path
#         self.preprocessor = TextPreprocessor()
#         self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
#         self.model = MultinomialNB()
#         self.X_train = None
#         self.X_test = None
#         self.y_train = None
#         self.y_test = None
    
#     def load_data(self):
#         """Load and preprocess the dataset"""
#         print("Loading dataset...")
#         df = pd.read_csv(self.dataset_path)
        
#         print(f"Dataset loaded: {len(df)} messages")
#         print(f"Spam messages: {sum(df['label'] == 'spam')}")
#         print(f"Ham messages: {sum(df['label'] == 'ham')}")
        
#         # Preprocess messages
#         print("\nPreprocessing messages...")
#         df['processed_message'] = self.preprocessor.preprocess_batch(df['message'].values)
        
#         # Convert labels to binary (0: ham, 1: spam)
#         df['label_binary'] = df['label'].map({'ham': 0, 'spam': 1})
        
#         return df
    
#     def split_data(self, df, test_size=0.2, random_state=42):
#         """Split data into training and testing sets"""
#         print(f"\nSplitting data (test size: {test_size})...")
#         X = df['processed_message']
#         y = df['label_binary']
        
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             X, y, test_size=test_size, random_state=random_state, stratify=y
#         )
        
#         print(f"Training set: {len(self.X_train)} messages")
#         print(f"Testing set: {len(self.X_test)} messages")
    
#     def vectorize_data(self):
#         """Convert text to TF-IDF features"""
#         print("\nVectorizing text data...")
#         self.X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
#         self.X_test_tfidf = self.vectorizer.transform(self.X_test)
        
#         print(f"Feature matrix shape: {self.X_train_tfidf.shape}")
    
#     def train_model(self):
#         """Train the Multinomial Naive Bayes classifier"""
#         print("\nTraining Multinomial Naive Bayes classifier...")
#         self.model.fit(self.X_train_tfidf, self.y_train)
#         print("Training completed!")
    
#     def evaluate_model(self):
#         """Evaluate model performance"""
#         print("\n" + "="*50)
#         print("MODEL EVALUATION")
#         print("="*50)
        
#         # Predictions
#         y_train_pred = self.model.predict(self.X_train_tfidf)
#         y_test_pred = self.model.predict(self.X_test_tfidf)
        
#         # Training accuracy
#         train_accuracy = accuracy_score(self.y_train, y_train_pred)
#         print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        
#         # Testing accuracy
#         test_accuracy = accuracy_score(self.y_test, y_test_pred)
#         print(f"Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
#         # Classification report
#         print("\nClassification Report:")
#         print(classification_report(self.y_test, y_test_pred, 
#                                    target_names=['Ham', 'Spam']))
        
#         # Confusion matrix
#         cm = confusion_matrix(self.y_test, y_test_pred)
#         print("\nConfusion Matrix:")
#         print(f"True Negatives (Ham): {cm[0][0]}")
#         print(f"False Positives (Ham classified as Spam): {cm[0][1]}")
#         print(f"False Negatives (Spam classified as Ham): {cm[1][0]}")
#         print(f"True Positives (Spam): {cm[1][1]}")
        
#         return test_accuracy
    
#     def save_model(self, model_dir='model'):
#         """Save trained model and vectorizer"""
#         print(f"\nSaving model to {model_dir}/...")
#         os.makedirs(model_dir, exist_ok=True)
        
#         # Save model
#         with open(f'{model_dir}/spam_classifier.pkl', 'wb') as f:
#             pickle.dump(self.model, f)
        
#         # Save vectorizer
#         with open(f'{model_dir}/tfidf_vectorizer.pkl', 'wb') as f:
#             pickle.dump(self.vectorizer, f)
        
#         print("Model and vectorizer saved successfully!")
    
#     def run_training_pipeline(self):
#         """Execute complete training pipeline"""
#         print("="*50)
#         print("SPAM CLASSIFIER TRAINING PIPELINE")
#         print("="*50)
        
#         # Load data
#         df = self.load_data()
        
#         # Split data
#         self.split_data(df)
        
#         # Vectorize
#         self.vectorize_data()
        
#         # Train
#         self.train_model()
        
#         # Evaluate
#         accuracy = self.evaluate_model()
        
#         # Save
#         self.save_model()
        
#         print("\n" + "="*50)
#         print("TRAINING COMPLETED SUCCESSFULLY!")
#         print("="*50)
        
#         return accuracy


# def main():
#     """Main function to run training"""
#     trainer = SpamClassifierTrainer()
#     trainer.run_training_pipeline()


# if __name__ == "__main__":
#     main()