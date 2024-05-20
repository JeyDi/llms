import spacy

# Load spaCy model
# Remember to download the model before running the code
# poetry run spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


# Function to clean text using spaCy
def clean_text(text):
    doc = nlp(text)
    clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(clean_tokens)
