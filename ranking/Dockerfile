FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip3 install -U scikit-learn

RUN pip3 install -U catboost

CMD ["python3", "main.py"]