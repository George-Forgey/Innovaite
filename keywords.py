from keybert import KeyBERT

# =============== KEYWORD + SENTIMENT EXTRACTORS ===============
# If you want a custom pipeline or keybert
kw_model = KeyBERT('sentence-transformers/all-MiniLM-L6-v2')

def get_keywords(text):
    # Extract top 5 keywords
    keyword_tuples = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2),
                                               stop_words='english', top_n=5)
    return [k for (k, s) in keyword_tuples]