import pandas as pd
import pickle
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load preprocessed data
X = load_npz('X.npz')
y = pd.read_csv('y.csv').values.ravel()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Save trained model
with open('spam_classifier.pkl', 'wb') as f:
    pickle.dump(nb, f)

# Evaluate model
y_pred = nb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))