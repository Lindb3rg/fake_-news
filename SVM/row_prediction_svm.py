# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# import spacy
# import string
# import joblib
# from dotenv import load_dotenv
# import os

# load_dotenv()

# SVM_VECTORIZER_LINK = os.environ.get('SVM_VECTORIZER_LINK')
# SVM_MODEL_LINK = os.environ.get('SVM_MODEL_LINK')

# # from links import link_svm, link_tfidf



# # Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Download spaCy model
# spacy.cli.download("en_core_web_sm")
# # Load spaCy model
# nlp = spacy.load('en_core_web_sm')



# loaded_tfidf_vectorizer = joblib.load(SVM_VECTORIZER_LINK)
# loaded_svm_model = joblib.load(SVM_MODEL_LINK)


# def svm_preprocess_text(text, use_lemmatization=True):
#     # Lowercasing
#     text = text.lower()
    
#     # Tokenization using spaCy
#     tokens = nlp(text)
#     tokens = [token.text for token in tokens]
    
#     # Punctuation Removal
#     tokens = [token for token in tokens if token not in string.punctuation]
    
#     # Stop Word Removal
#     stop_words = set(stopwords.words('english'))
#     tokens = [token for token in tokens if token not in stop_words]
    
#     # Stemming
#     stemmer = PorterStemmer()
#     tokens = [stemmer.stem(token) for token in tokens]
    
#     # Lemmatization (optional)
#     if use_lemmatization:
#         lemmatizer = WordNetLemmatizer()
#         tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
#     return tokens


# def svm_label_text(original_text):
#     if not isinstance(original_text, list):
#         original_text = [original_text]
    
#     preprocessed_texts = [svm_preprocess_text(x) for x in original_text]
#     processed_text_strings = [' '.join(text) for text in preprocessed_texts]
#     tfidf_features = loaded_tfidf_vectorizer.transform(processed_text_strings)
#     predictions = loaded_svm_model.predict(tfidf_features)
    
#     threshold = 0.5
#     binary_predictions = (predictions >= threshold).astype(int)
    
#     list_of_predictions = []
#     # Create dict with prediction
#     for i in binary_predictions:
#         if i == 1:
#             list_of_predictions.append("Fake")
#         else:
#             list_of_predictions.append("True")

#     return list_of_predictions
