import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import re
from scipy.sparse import hstack



df = pd.read_csv('enron_spam_data.csv') #Getting the training data from the csv

#preprocess the data before deploying a model 
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#function which generate the features
def pre_process(text):
    tokens = word_tokenize(text.lower()) #convert the text to all lowercase words and then tokenize 

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words] # store the lemmatized version of all the words in text (excluding words that aren't strictly alphabetical and words within stop_words  
    return ' '.join(tokens)

#appending the feature columns to our dataframe
df['Message'] = df['Message'].fillna('') #fill N/A messages with empty string 
df['Cleaned_Message'] = df['Message'].apply(pre_process) #Create the cleaned_message feature 
df['Spam/Ham'] = df['Spam/Ham'].map({'spam': 0, 'ham': 1}) #Binarize the data so that Spam = 0 and Ham = 1
df = df.dropna(subset=['Spam/Ham']) #drop any rows which don't have a label 


print('finished cleaning the data')
#df.to_csv('cleaned_enron_data.csv')

#Now have cleaned data, can move onto  

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Message']) #Perform TF-IDF on the cleaned message to get a numerical representation 
y = df['Spam/Ham']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #split the data into training and testing sets


#now time to choose a model 

nb = MultinomialNB()

#train model on training data 
nb.fit(X_train, y_train)



#now we can evaluate the model 
y_pred = nb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))