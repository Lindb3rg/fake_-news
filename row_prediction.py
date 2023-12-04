# Imports

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import spacy
import pickle
 
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Download spaCy model
spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")


# Padding sequences
maxlen = 75

# load model
loaded_model = load_model("trained_models/keras_model.h5")

loaded_tokenizer = Tokenizer()

# Load the tokenizer from the file
with open('tokenizer/tokenizer.pkl', 'rb') as tokenizer_file:
    loaded_tokenizer = pickle.load(tokenizer_file)


# preprocess function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    
    # Tokenization using spaCy
    tokens = nlp(text)
    tokens = [token.text for token in tokens]
    
    # Punctuation Removal
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Stop Word Removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens


def label_text(original_text):
    preprocessed_texts = preprocess_text(original_text)
    new_sequences = loaded_tokenizer.texts_to_sequences(preprocessed_texts)
    new_padded_sequence = pad_sequences(new_sequences, maxlen=maxlen)
    predictions = loaded_model.predict(new_padded_sequence)
    
    threshold = 0.5
    binary_predictions = (predictions >= threshold).astype(int)

    # Create dict with prediction
    dict = {"Headline": original_text, "Outcome": ["predicted as fake" if pred == 1 else "predicted as real" for pred in binary_predictions]}
    
    return dict