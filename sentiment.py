import torch
from transformers import pipeline

def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    sen_value = sentiment_pipeline(text)
    # Return both the label and the score
    return sen_value[0]["label"], sen_value[0]["score"]