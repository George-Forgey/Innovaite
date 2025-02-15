import torch
from transformers import pipeline

def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    sen_value = sentiment_pipeline(text)
    return sen_value[0]["score"]