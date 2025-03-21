import pickle
import pandas as pd
from scipy.sparse import hstack, load_npz

# Load vectorizer and model
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('spam_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Sample input
new_message = ["Congratulations! You've won a free iPhone!"]

# Preprocess (apply the same cleaning process as before)
cleaned_message = [' '.join(vectorizer.build_analyzer()(msg)) for msg in new_message]

# Transform input using the saved TF-IDF vectorizer
X_input = vectorizer.transform(cleaned_message)

# Predict
prediction = model.predict(X_input)
result = "Ham" if prediction[0] == 1 else "Spam"
print(f"Prediction: {result}")