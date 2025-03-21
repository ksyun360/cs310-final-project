import json
import re
import boto3
import uuid
import tarfile
import os
import sklearn
import pickle

runtime = boto3.client("sagemaker-runtime")
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("SpamDetectionLogs")
s3 = boto3.client ('s3')

def clean_text(text): 
    """Basic text preprocessing: remove special characters, convert to lowercase."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

BUCKET_NAME = 'nu-cs310-sbascoe23'
MODEL_FILE = 'models/model.tar.gz'

# Load model and vectorizer globally to reuse across invocations
model = None
vectorizer = None

def load_model():
    global model, vectorizer
    
    # Download the model from S3
    model_path = '/tmp/model.tar.gz'
    s3.download_file(BUCKET_NAME, MODEL_FILE, model_path)
    
    # Extract the tar.gz file
    with tarfile.open(model_path, 'r:gz') as tar:
        tar.extractall(path='/tmp')
    
    # Load model and vectorizer using pickle
    with open('/tmp/spam_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('/tmp/tfidf_vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        email_subject = body.get("subject", "")
        email_body = body.get("message", "")


        # Preprocess inputs
        cleaned_subject = clean_text(email_subject)
        cleaned_body = clean_text(email_body)
        combined_text = f"{cleaned_subject} {cleaned_body}"

        global model, vectorizer
    
        if model is None or vectorizer is None:
            load_model()

        input_vector = vectorizer.transform([combined_text])
        
        # Make prediction
        prediction = int(model.predict(input_vector)[0])
        if prediction == 0:
            result = "spam"
        else:
            result = "ham"

        messageid = str(uuid.uuid4())

        table.put_item(
            Item={
                "messageid": messageid,
                "subject": email_subject,
                "body": email_body,
                "prediction": result
            }
        )

        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "OPTIONS, POST, GET",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": json.dumps({"messageid": messageid, "prediction": result})
        }
    except Exception as err:
        print("**ERROR**")
        print(str(err))
        
        return {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "OPTIONS, POST, GET",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": json.dumps({"error": str(err)})
        }
