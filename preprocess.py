import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean text
def pre_process(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Load data
df = pd.read_csv('enron_spam_data.csv')
df['Message'] = df['Message'].fillna('')
df['Cleaned_Message'] = df['Message'].apply(pre_process)
df['Spam/Ham'] = df['Spam/Ham'].map({'spam': 0, 'ham': 1})
df = df.dropna(subset=['Spam/Ham'])

# Save cleaned data
df.to_csv('cleaned_enron_data.csv', index=False)

# Fit TF-IDF vectorizer and save it
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Message'])

# Save vectorizer and transformed data
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save transformed data and labels
import scipy.sparse
scipy.sparse.save_npz('X.npz', X)
df['Spam/Ham'].to_csv('y.csv', index=False)

print('Preprocessing complete. Cleaned data saved.')