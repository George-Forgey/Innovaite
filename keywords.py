from keybert import KeyBERT

# Initialize KeyBERT with our model
kw_model = KeyBERT('sentence-transformers/all-MiniLM-L6-v2')

# Extract keywords and return only the keyword list
def get_keywords(text):
    # Extract keywords; this returns a list of tuples like [(keyword, score), ...]
    keyword_tuples = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    # Extract just the keyword strings
    keywords = [keyword for keyword, score in keyword_tuples]
    return keywords
