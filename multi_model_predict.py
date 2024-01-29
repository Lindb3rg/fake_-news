from model import MultiPrediction,FilePrediction
from single_model_predict import preprocess_text, loaded_logistic_model,loaded_sequential_model,\
    loaded_svm_model,loaded_tfidf_vectorizer,loaded_tokenizer

from formatting_functions import format_float
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt



def preprocess_for_specific_models(original_text: list, models:list = None)-> dict:
    
    
    preprocessed_texts = [preprocess_text(x) for x in original_text]

    maxlen = 75
    
    text_processed_for_models = {}
    
    for model in models:
        
        if model == "svm" or model == "logistic":
            processed_text_strings = [' '.join(text) for text in preprocessed_texts]
            tfidf_features = loaded_tfidf_vectorizer.transform(processed_text_strings)
            text_processed_for_models[model] = tfidf_features

        elif model == "sequential":
            new_sequences = loaded_tokenizer.texts_to_sequences(preprocessed_texts)
            new_padded_sequence = pad_sequences(new_sequences, maxlen=maxlen)
            text_processed_for_models[model] = new_padded_sequence

        else:
            print("Error in preprocessing")
        
    
    return text_processed_for_models


def voter(true: tuple,fake: tuple)->tuple:
    prediction = [true,fake]
    return max(prediction, key=lambda x: x[1])



def manually_classify_vote(percentage:float, binary:int):
    
    if binary == 0:
        return ("True",(100 - percentage))

    return ("Fake",(100 - percentage))
    
    
    
    


def multi_model_predict_text(texts: list|str,input_type:str,**kwargs)-> MultiPrediction | FilePrediction:
    
    if not isinstance(texts, list):
        texts = [texts]
    

    
    model_names = ["logistic","svm","sequential"]
                        
        
    
    file_name = kwargs.get("file_name")
    new_id = kwargs.get("new_id")
    new_group_id = kwargs.get("new_group_id")
    
    
    
    dict_of_preprocessed_texts = preprocess_for_specific_models(original_text=texts,
                                                                models=model_names)
   
    logistic_predictions = loaded_logistic_model.predict(dict_of_preprocessed_texts["logistic"])
    logistic_row_predictions = loaded_logistic_model.predict_proba(dict_of_preprocessed_texts["logistic"])
    logistic_predictions = logistic_predictions.reshape(-1, 1)
    
    svm_predictions = loaded_svm_model.predict(dict_of_preprocessed_texts["svm"])
    svm_row_predictions = loaded_svm_model.predict_proba(dict_of_preprocessed_texts["svm"])
    svm_predictions = svm_predictions.reshape(-1, 1)
    
    sequential_predictions = loaded_sequential_model.predict(dict_of_preprocessed_texts["sequential"])
    sequential_predictions = sequential_predictions.reshape(-1, 1)
    sequential_predictions_binary = (sequential_predictions > 0.5).astype(int)
    
    


    
    if file_name:
        file_object = FilePrediction()
        file_object.group_id = new_group_id
        file_object.file_name = file_name
        file_object.input_type = "file"
        file_object.identity = "multi"
        file_object.prediction_result_for_each_model = [("Logistic",logistic_predictions),
                                                        ("SVM",svm_predictions),
                                                        ("Sequential",sequential_predictions_binary)]
        
    
    
    for index in range(len(texts)):
    
            prediction_object = MultiPrediction()
            prediction_object.model_selected = "all_models"
            prediction_object.text = texts[index]
            prediction_object.input_type = "text"
            
            true = ("True",(svm_row_predictions[index][0]*100))
            fake = ("Fake",(svm_row_predictions[index][1]*100))
            vote = voter(true,fake)
            prediction_object.set_model_vote("svm", (vote[0],format_float(vote[1])))
            

            true = ("True",(logistic_row_predictions[index][0]*100))
            fake = ("Fake",(logistic_row_predictions[index][1]*100))            
            vote = voter(true,fake)
            prediction_object.set_model_vote("logistic", (vote[0],format_float(vote[1])))
            
            
            manual_vote = manually_classify_vote(sequential_predictions[[index]],sequential_predictions_binary[index])
            prediction_object.set_model_vote("sequential", (manual_vote[0],format_float(float(manual_vote[1]))))

            if file_name:
                file_object.add_prediction_object(prediction_object)
                prediction_object.id = index
            else:
                prediction_object.id = new_id
    
    
    
            prediction_object.get_majority_prediction()
    
    if not file_name:
        
        
        return prediction_object
    
    
    file_object.file_total_score()
    file_object.bar_plot_for_file()
    file_object.bar_plot_all_predictions()
    file_object.draw_bar_plots()
    
    return file_object



    



