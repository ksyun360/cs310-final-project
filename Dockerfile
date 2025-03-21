FROM public.ecr.aws/lambda/python:3.11

WORKDIR /var/task

COPY app.py ./
COPY requirements.txt ./

RUN pip install -r requirements.txt

CMD ["app.lambda_handler"]