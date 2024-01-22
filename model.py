import datetime
import matplotlib.pyplot as plt
import numpy as np 
plt.switch_backend('Agg')
from typing import Iterator




class SinglePrediction:
    def __init__(self, text:str=None,prediction:str=None,accuracy:float=None,model_selected:str=None):
        self.id = int
        self.text = text
        self.prediction = prediction
        self.accuracy = accuracy
        self.model_selected = model_selected
        self.identity = "single"
        self.date = datetime.datetime.now()
        self.image_name = str
        self.input_type = str

    def create_histogram(self):
        model = [self.model_selected]
        
        fig, ax = plt.subplots(figsize=(10, 8))

        color_theme = ['skyblue', 'salmon', 'lightgreen', 'gold']

        ax.bar(model, self.accuracy, color=color_theme, edgecolor='black')
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_title('Probability Histogram')
        ax.set_yticks(np.arange(0, 101, 5))
        ax.grid(True)

        
        date = str(self.date).replace(" ", "_")
        save_path = f"static/model_images/single_model_images/{self.model_selected}_histogram_{date}.png"
        self.image_name = f"{self.model_selected}_histogram_{date}.png"
        fig.savefig(save_path, format='png')
        return
    
    def get_histogram_image(self):
        img_dir = "static/model_images/single_model_images/"
        return f"{img_dir}{self.image_name}"

    def get_text(self):
        return self.text
    
    def get_prediction(self):
        return self.prediction

    def get_accuracy(self):
        return self.accuracy
    
    def get_model_selected(self):        
        return self.model_selected
    
    def get_identity(self):
        return self.identity
    
    def get_date(self):
        return self.date
    
    def get_input_type(self):
        return self.input_type
    
    def get_id(self):
        return self.id  


    

class MultiPrediction:

    def __init__(self, text:str=None,prediction:str=None,accuracy:float=None,model_selected:str=None):
        self.id = int
        self.text = text
        self.prediction = prediction
        self.accuracy = accuracy
        self.model_selected = model_selected
        self.model = {"logistic":("vote","probability"),
                      "svm":("vote","probability"),
                      "sequential":("vote","probability")}
        
        self.identity = "multi"
        self.date = datetime.datetime.now()
        self.image_name = str
        self.input_type = str

    def _collect_probabilities(self):
        probabilities = []
        probabilities = [self.model[model][1] for model in self.model]
        probabilities.append(self.accuracy)
        return probabilities 
    
    def get_histogram_image(self):
        img_dir = "static/model_images/all_models_images/"
        return f"{img_dir}{self.image_name}"



    def create_histogram(self):
        models = ["logistic","svm","sequential","all_models"]
        model_probabilities = self._collect_probabilities()
        
        fig, ax = plt.subplots(figsize=(10, 8))

        color_theme = ['skyblue', 'salmon', 'lightgreen', 'gold']

        ax.bar(models, model_probabilities, color=color_theme, edgecolor='black')
        ax.set_xlabel('Models')
        ax.set_ylabel('Probability')
        ax.set_title('Probability Histogram')
        ax.set_yticks(np.arange(0, 101, 5))
        ax.grid(True)

        # date = self.date
        date = str(self.date).replace(" ", "_")
        save_path = f"static/model_images/all_models_images/histogram_{date}.png"
        self.image_name = f"histogram_{date}.png"
        fig.savefig(save_path, format='png')
        return



    def get_identity(self):
        return self.identity

    def get_text(self):
        return self.text
    
    def get_prediction(self):
        return self.prediction

    def get_accuracy(self):
        return self.accuracy
    
    def get_model_selected(self):        
        return self.model_selected

    def get_model_vote(self, model:str):
        if model in self.model:
            return self.model[model]
    
    def get_date(self):
        return self.date
    
    def set_model_vote(self, model:str, vote_tuple:tuple):
        self.model[model] = vote_tuple

    def set_image_link(self, link:str):
        self.image_link = link

    def get_input_type(self):
        return self.input_type
    
    def get_id(self):
        return self.id


class FilePrediction:
    def __init__(self, file_name:str=None):
        self.group_id = int
        self.file_name = file_name
        self.prediction_objects = []
        self.identity = str
        self.date = datetime.datetime.now()
        self.input_type = str
        
    def get_identity(self):
        return self.identity
    
    def add_prediction_object(self, prediction_object:SinglePrediction):
        self.prediction_objects.append(prediction_object)

    def get_file_name(self):
        return self.file_name
    
    def get_input_type(self):
        return self.input_type
    
    def get_group_id(self):
        return self.group_id
    

    def get_all_prediction_objects(self):
        return self.prediction_objects
    




if __name__ == "__main__":

    prediction = MultiPrediction()
    print(prediction.get_identity())