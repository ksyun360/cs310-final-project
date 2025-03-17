

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import re
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier



df = pd.read_csv('enron_spam_data.csv') #Getting the training data from the csv

#preprocess the data before deploying a model 
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#Function which generate the features
def pre_process(text):
    tokens = word_tokenize(text.lower()) #convert the text to all lowercase words and then tokenize 

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words] # store the lemmatized version of all the words in text (excluding words that aren't strictly alphabetical and words within stop_words  
    return ' '.join(tokens)

def count_capitals(text):
    return sum(1 for c in text if c.isupper())

def count_punctuation(text):
    return sum(1 for c in text if c in string.punctuation)

def count_urls(text):
    return len(re.findall(r'(https?://\S+)', text))

#Appending the feature columns to our dataframe
df['Message'] = df['Message'].fillna('') #fill N/A messages with empty string 
df['Cleaned_Message'] = df['Message'].apply(pre_process) #Create the cleaned_message feature 
df['Spam/Ham'] = df['Spam/Ham'].map({'spam': 0, 'ham': 1}) #Binarize the data so that Spam = 0 and Ham = 1
df = df.dropna(subset=['Spam/Ham']) #drop any rows which don't have a label 
df['num_capitals'] = df['Message'].apply(count_capitals) #Create the num_capitals feature
df['num_punctuation'] = df['Message'].apply(count_punctuation) #Create the num_punctuation feature 
df['num_urls'] = df['Message'].apply(count_urls) #Create the num_urls feature 
df['num_words'] = df['Message'].apply(lambda x: len(x.split())) #Create the num_words feature 
df['message_length'] = df['Message'].apply(len) #Create the message_length feature (different from num_words - counts total characters)



print('finished cleaning the data/creating the necessary features')
#df.to_csv('cleaned_enron_data.csv')

#Now have cleaned data, can move onto  

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_tfidf = vectorizer.fit_transform(df['Cleaned_Message']) #Perform TF-IDF on the cleaned message to get a numerical representation 


# Scale numeric features to keep them in a consistent range
scaler = MinMaxScaler() #Squeezes values into a range [0,1]
scaled_features = scaler.fit_transform(df[['num_capitals', 'num_punctuation', 'num_words', 'message_length', 'num_urls']])

X_features = hstack([
    X_tfidf, 
    scaled_features
])  #Store the features which we will use to train the model 
y = df['Spam/Ham'] 

X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42) #split the data into training and testing sets


# Convert sparse matrix to dense since XGBoost requires dense input
X_train = X_train.toarray()
X_test = X_test.toarray()

#Now time to choose a model 

xgb = XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=200,
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), # Ratio of ham to spam
    min_child_weight=3,
    subsample=0.9,
    colsample_bytree=0.9,
    use_label_encoder=False,
    eval_metric='logloss'
)

#fit the model to the training data
xgb.fit(X_train, y_train)


#Now we can evaluate the model 
y_pred = xgb.predict(X_test)

print('using XGBoost!')
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
