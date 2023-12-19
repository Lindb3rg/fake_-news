import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
import string
import joblib
import pickle
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from dotenv import load_dotenv


is_heroku = 'DYNO' in os.environ

if is_heroku:
    SVM_VECTORIZER_LINK = os.environ.get('SVM_VECTORIZER_LINK')
    SVM_MODEL_LINK = os.environ.get('SVM_MODEL_LINK')
    LOGISTIC_MODEL_LINK = os.environ.get('LOGISTIC_MODEL_LINK')
    SEQUENTIAL_MODEL_LINK = os.environ.get('SEQUENTIAL_MODEL_LINK')
    SEQUENTIAL_TOKENIZER_LINK = os.environ.get('SEQUENTIAL_TOKENIZER_LINK')
else:
    load_dotenv()
    SVM_VECTORIZER_LINK = os.environ.get('SVM_VECTORIZER_LINK')
    SVM_MODEL_LINK = os.environ.get('SVM_MODEL_LINK')
    LOGISTIC_MODEL_LINK = os.environ.get('LOGISTIC_MODEL_LINK')
    SEQUENTIAL_MODEL_LINK = os.environ.get('SEQUENTIAL_MODEL_LINK')
    SEQUENTIAL_TOKENIZER_LINK = os.environ.get('SEQUENTIAL_TOKENIZER_LINK')


loaded_tokenizer = Tokenizer()


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")

maxlen = 75


loaded_sequential_model = load_model(SEQUENTIAL_MODEL_LINK)
loaded_tfidf_vectorizer = joblib.load(SVM_VECTORIZER_LINK)
loaded_svm_model = joblib.load(SVM_MODEL_LINK)
loaded_logistic_model = joblib.load(LOGISTIC_MODEL_LINK)
loaded_tokenizer = Tokenizer()


with open(SEQUENTIAL_TOKENIZER_LINK, 'rb') as tokenizer_file:
    loaded_tokenizer = pickle.load(tokenizer_file)



def preprocess_text(text:list)->list[str]:
    
    text = text.lower()
    
    tokens = nlp(text)
    tokens = [token.text for token in tokens]
    
    tokens = [token for token in tokens if token not in string.punctuation]
    
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens





def model_predict_text(original_text:list, modelname="svm")->dict:
    if not isinstance(original_text, list):
        original_text = [original_text]

    preprocessed_texts = [preprocess_text(x) for x in original_text]

    predictions = ""
    loaded_model = ""

    if modelname == "all_models":
        predictions = all_models_predict_text(preprocessed_texts, original_text)
        return predictions
    
    elif modelname == "svm":
        loaded_model = loaded_svm_model

    elif modelname == "logistic":
        loaded_model = loaded_logistic_model

    elif modelname == "sequential":
        loaded_model = loaded_sequential_model


    if modelname == "svm" or modelname == "logistic":
        processed_text_strings = [' '.join(text) for text in preprocessed_texts]
        tfidf_features = loaded_tfidf_vectorizer.transform(processed_text_strings)
        predictions = loaded_model.predict(tfidf_features)

    elif modelname == "sequential":
        new_sequences = loaded_tokenizer.texts_to_sequences(preprocessed_texts)
        new_padded_sequence = pad_sequences(new_sequences, maxlen=maxlen)
        predictions = loaded_model.predict(new_padded_sequence)

    threshold = 0.5
    binary_predictions = (predictions >= threshold).astype(int)
    
    dict_of_predictions = {}
    
    for index, i in enumerate(binary_predictions):
        if i == 1:
            key = str(original_text[index])
            dict_of_predictions[key] = "Fake"
        else:
            key = str(original_text[index])
            dict_of_predictions[key] = "True"


    return dict_of_predictions
    





def all_models_predict_text(preprocessed_texts:list[str], original_text:list)->dict:

    processed_text_strings = [' '.join(text) for text in preprocessed_texts]
    tfidf_features = loaded_tfidf_vectorizer.transform(processed_text_strings)

    new_sequences = loaded_tokenizer.texts_to_sequences(preprocessed_texts)
    new_padded_sequence = pad_sequences(new_sequences, maxlen=maxlen)

    logistic_predictions = loaded_logistic_model.predict(tfidf_features)
    svm_predictions = loaded_svm_model.predict(tfidf_features)
    sequential_predictions = loaded_sequential_model.predict(new_padded_sequence)

    threshold = 0.5
    logistic_binary_predictions = (logistic_predictions >= threshold).astype(int)
    svm_binary_predictions = (svm_predictions >= threshold).astype(int)
    sequential_binary_predictions = (sequential_predictions >= threshold).astype(int)

    final_predictions = {}

    votes_for_fake = 0

    for index in range(len(logistic_binary_predictions)):
        if logistic_binary_predictions[index] == 1:
            votes_for_fake +=1

        if svm_binary_predictions[index] == 1:
            votes_for_fake +=1

        if sequential_binary_predictions[index] == 1:
            votes_for_fake +=1

        if votes_for_fake > 1:
            key = str(original_text[index])
            final_predictions[key] = "Fake"
        else:
            key = str(original_text[index])
            final_predictions[key] = "True"
        
        votes_for_fake = 0
    

    return final_predictions






