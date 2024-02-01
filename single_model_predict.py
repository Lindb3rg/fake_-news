import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
import string
import joblib
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import os
from dotenv import load_dotenv
from model import FilePrediction, SinglePrediction
from formatting_functions import convert_to_binary, format_float


"""
Create own .env file with the following variables and their paths:

"""
load_dotenv()
SVM_VECTORIZER_LINK = os.environ.get('SVM_VECTORIZER_LINK')
SVM_MODEL_LINK = os.environ.get('SVM_MODEL_LINK')
LOGISTIC_MODEL_LINK = os.environ.get('LOGISTIC_MODEL_LINK')
SEQUENTIAL_MODEL_LINK = os.environ.get('SEQUENTIAL_MODEL_LINK')
SEQUENTIAL_TOKENIZER_LINK = os.environ.get('SEQUENTIAL_TOKENIZER_LINK')

loaded_tokenizer = Tokenizer()

try:
    stop_words = set(stopwords.words('english'))
    print("Found!")
except LookupError:
    print("Not found. Downloading stopwords...")
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm...")
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

def preprocess_text(text: list) -> list[str]:
    text = text.lower()
    tokens = nlp(text)
    tokens = [token.text for token in tokens]
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def single_model_predict_text(original_text: list, modelname="svm", **kwargs) -> SinglePrediction | FilePrediction:
    if not isinstance(original_text, list):
        original_text = [original_text]
        
    file_name = kwargs.get("file_name")
    new_id = kwargs.get("new_id")
    new_group_id = kwargs.get("new_group_id")
    input_type = kwargs.get("input_type")
    
    if input_type == "api":
        new_id = "api"
        
    preprocessed_texts = [preprocess_text(x) for x in original_text]
    predictions = ""
    loaded_model = ""
    row_predictions = []
    
    if modelname == "svm":
        loaded_model = loaded_svm_model
    elif modelname == "logistic":
        loaded_model = loaded_logistic_model
    elif modelname == "sequential":
        loaded_model = loaded_sequential_model
    else:
        raise ValueError("Invalid modelname provided.")
    
    if modelname == "svm" or modelname == "logistic":
        processed_text_strings = [' '.join(text) for text in preprocessed_texts]
        tfidf_features = loaded_tfidf_vectorizer.transform(processed_text_strings)
        row_predictions = prediction_probability_per_row(loaded_model, tfidf_features, modelname)
        predictions = loaded_model.predict(tfidf_features)
    elif modelname == "sequential":
        new_sequences = loaded_tokenizer.texts_to_sequences(preprocessed_texts)
        new_padded_sequence = pad_sequences(new_sequences, maxlen=maxlen)
        row_predictions = prediction_probability_per_row(loaded_model, new_padded_sequence, modelname)
        predictions = loaded_model.predict(new_padded_sequence)
    binary_predictions = convert_to_binary(predictions, threshold=0.5)
    
    if file_name:
        file_prediction_object = FilePrediction()
        file_prediction_object.group_id = new_group_id
        file_prediction_object.file_name = file_name
        file_prediction_object.input_type = "file"
        file_prediction_object.identity = "single"
        
    for index, i in enumerate(binary_predictions):
        prediction_object = SinglePrediction()
        if i == 1:
            key = str(original_text[index])
            prediction_object.text = key
            prediction_object.prediction = "Fake"
        else:
            key = str(original_text[index])
            prediction_object.text = key
            prediction_object.prediction = "True"
        accuracy = row_predictions[index]
        prediction_object.accuracy = accuracy
        prediction_object.model_selected = modelname
        prediction_object.identity = "single"
        
        if input_type == "api":
            prediction_object.input_type = input_type
        else:
            prediction_object.input_type = "text"
            
        if file_name:
            file_prediction_object.add_prediction_object(prediction_object)
            prediction_object.id = index
        else:
            prediction_object.id = new_id
    if file_name:
        return file_prediction_object
    return prediction_object

def prediction_probability_per_row(model, text_processor, modelname):
    if modelname == "sequential":
        row_pred_seq = []
        y_pred = model.predict(text_processor)
        y_pred_binary = (y_pred > 0.5).astype(int)
        probabilities = y_pred
        for i, probability in enumerate(probabilities[:, 0]):
            probability *= 100
            pred = y_pred_binary[i]
            if pred == 0:
                probability = 100 - probability
            probability = format_float(probability)
            probabilities = probability
            row_pred_seq.append(probabilities)
        return row_pred_seq
    
    probabilities = model.predict_proba(text_processor)
    row_pred = []
    
    for i, (probability_0, probability_1) in enumerate(zip(probabilities[:, 0], probabilities[:, 1])):
        if probability_1 >= 0.5:
            prob = (probability_1 * 100)
            prob = format_float(prob)
            row_pred.append(prob)
        else:
            prob = probability_0 * 100
            prob = format_float(prob)
            row_pred.append(prob)
            
    return row_pred
