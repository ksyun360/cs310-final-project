import json
import re
import boto3
import uuid

runtime = boto3.client("sagemaker-runtime")
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("SpamDetectionLogs")

def clean_text(text): #kinda here as placeholder, can make call to Sean's preprocessing
    """Basic text preprocessing: remove special characters, convert to lowercase."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        email_subject = body.get("subject", "")
        email_body = body.get("message", "")

        # Preprocess inputs
        cleaned_subject = clean_text(email_subject) #or make call to Sean's preprocessing
        cleaned_body = clean_text(email_body)
        combined_text = f"{cleaned_subject} {cleaned_body}"

        # Convert to SageMaker-compatible format
        payload = combined_text.encode("utf-8")

        # Call SageMaker endpoint
        # response = runtime.invoke_endpoint(
        #     EndpointName="OUR-SAGEMAKER-ENDPOINT",
        #     ContentType="text/csv",
        #     Body=payload
        # )

        # result = json.loads(response["Body"].read().decode())
        result = "spam" #placeholder for now
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