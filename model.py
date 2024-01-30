import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
import os

from formatting_functions import format_float
plt.switch_backend('Agg')




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
        date = str(self.date).replace(":", "-")
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
        self.bar_blot_image_name = str
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
        date = str(self.date).replace(":", "-")
        save_path = f"static/model_images/all_models_images/histogram_{date}.png"
        self.image_name = f"histogram_{date}.png"
        fig.savefig(save_path, format='png')
        return
    
    
    
    
    
        
    
    
    def get_majority_prediction(self):
    
        true = 0
        true_total = 0
        fake = 0
        fake_total = 0
        
        
        for value in self.model.values():
            if value[0] == "True":
                true += 1
                true_total += value[1]
            else:
                fake += 1
                fake_total += value[1]
        
        if true > fake:
            self.prediction = "True"
            self.accuracy = format_float((true_total/true))
            
            
        elif true < fake:
            self.prediction = "Fake"
            self.accuracy = format_float((fake_total/fake))

    def get_prediction_for_final_barplot(self):
        return self.prediction,self.accuracy

    
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
    
    def get_all_votes(self):
        list_of_votes = []
        for model in self.model:
            list_of_votes.append(0 if self.model[model][0]=="True" else 1)
            
        list_of_votes.append(0 if self.prediction == "True" else 1)
        return list_of_votes




class FilePrediction:
    def __init__(self, file_name:str=None):
        self.group_id = int
        self.file_name = file_name
        self.prediction_objects = []
        self.identity = str
        self.date = datetime.datetime.now()
        self.input_type = str
        self.prediction_result_for_each_model = [tuple]
        self.file_total_true_count = 0
        self.file_total_fake_count = 0
        self.file_total_true_percentage = 0
        self.file_total_fake_percentage = 0
        self.average_for_true_in_file = 0
        self.average_for_fake_in_file = 0
        self.barplot_file_name = ""
        self.barplot_name = ""
        self.barplot_file_all_images = ""
        
        
        
        
    def get_identity(self)-> str:
        return self.identity
    
    def add_prediction_object(self, prediction_object:SinglePrediction)->None:
        self.prediction_objects.append(prediction_object)

    def get_file_name(self)-> str:
        return self.file_name
    
    def get_input_type(self)-> str:
        return self.input_type
    
    def get_group_id(self)-> int:
        return self.group_id
    
    def file_total_score(self)->None:
        
        for object in self.prediction_objects:

            if object.get_prediction_for_final_barplot()[0] == ("True"):
                
                self.file_total_true_count += 1
                
            else:
                self.file_total_fake_count += 1
                
        
        self._calculate_percentage()
        
    
    
    def _calculate_average(self)->None:
        self.average_for_true_in_file = self.file_total_true_percentage/self.file_total_true_count
        self.average_for_fake_in_file = self.file_total_fake_percentage/self.file_total_fake_count
    
    
    def _calculate_percentage(self)->None:
        total = self.file_total_true_count + self.file_total_fake_count
        self.file_total_true_percentage = (self.file_total_true_count/total)*100
        self.file_total_fake_percentage = (self.file_total_fake_count/total)*100
    
    def get_predition_counts(self)-> dict:
        return {"True":self.file_total_true_count, "Fake":self.file_total_fake_count}
    

    def get_all_prediction_objects(self)->[MultiPrediction]:
        return self.prediction_objects
    
    def get_all_barplots(self):
        return f"static/model_images/all_models_images/{self.barplot_file_all_images}"
        
    
    def _collect_binary_predictions(self)->None:
        
        binary_predictions = []
        for object in self.prediction_objects:

            object_votes = object.get_all_votes()

            binary_predictions.append(object_votes)
            
        return binary_predictions
    
    
    def summerize_images(self,img_1,img_2):
        
        # reading images 
        Image1 = plt.imread(img_1) 
        Image2 = plt.imread(img_2) 
        
        
        
        fig = plt.figure(figsize=(15,10.5)) 
  
        # setting values to rows and column variables 
        rows = 2
        columns = 1
        
        
        
        
        # Adds a subplot at the 1st position 
        fig.add_subplot(rows, columns, 1) 
        
        # showing image 
        plt.imshow(Image1) 
        plt.axis('off') 
        plt.title("Barplot per Model") 
        
        # Adds a subplot at the 2nd position 
        fig.add_subplot(rows, columns, 2) 
        
        # showing image 
        plt.imshow(Image2) 
        plt.axis('off') 
        plt.title("Barplot for File") 
        
       

        # Save the merged image
        date = datetime.datetime.now()
        date = str(self.date).replace(" ", "_")
        date = date.replace(":", "-")
        save_path = f"static/model_images/all_models_images/barplot_file_all_images_{date}.png"
        self.barplot_file_all_images = f"barplot_file_all_images_{date}.png"
        plt.savefig(save_path, format='png',bbox_inches='tight', pad_inches=0)
        return
        

        
        
    
    
    
    def draw_bar_plots(self)->None:

        plot_1 = f"static/model_images/all_models_images/{self.barplot_name}"
        plot_2 = f"static/model_images/all_models_images/{self.barplot_file_name}"
        if self.barplot_file_all_images == "":
            self.summerize_images(plot_1,plot_2)
        
            
            
        
    
         
        
    

    def bar_plot_all_predictions(self):
        
        
        model_names = [model[0] for model in self.prediction_result_for_each_model]

        models_prediction_result = [model[1] for model in self.prediction_result_for_each_model]
        
        all_predictions = np.hstack(models_prediction_result)

        gold_color = (1.0, 0.84, 0.0)
        dark_blue_color = (0.0, 0.0, 0.5)

        # Count the occurrences of each prediction for each model
        counts_0 = np.sum(all_predictions == 0, axis=0)
        counts_1 = np.sum(all_predictions == 1, axis=0)
        total_instances = len(all_predictions)

        # Calculate percentages
        percentage_0 = (counts_0 / total_instances) * 100
        percentage_1 = (counts_1 / total_instances) * 100

        # Set the figure size and style
        # sns.set(style="whitegrid")  # Use a seaborn style
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotting the grouped bar chart
        width = 0.35  # Width of each bar
        ind = np.arange(len(model_names))

        bars_0 = ax.bar(ind - width/2, percentage_0, width, label='Predicted 0', color=gold_color)
        bars_1 = ax.bar(ind + width/2, percentage_1, width, label='Predicted 1', color=dark_blue_color)

        ax.set_xlabel('Model', fontsize=14)
        ax.set_ylabel('Percentage', fontsize=14)
        ax.set_title('Percentage of Predictions for True and Fake by All Models', fontsize=16)
        ax.set_xticks(ind)
        ax.set_xticklabels(model_names, fontsize=12)
        ax.legend(fontsize=12)

        # Add percentage labels on top of each bar with a different color
        for bar_0, bar_1 in zip(bars_0, bars_1):
            height_0 = bar_0.get_height()
            height_1 = bar_1.get_height()
            
            ax.annotate(f'{height_0:.2f}%', xy=(bar_0.get_x() + bar_0.get_width() / 2, height_0),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color='black', fontsize=10)
            
            ax.annotate(f'{height_1:.2f}%', xy=(bar_1.get_x() + bar_1.get_width() / 2, height_1),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color='black', fontsize=10)

        # Set the y-axis scale to show 0-100%
        ax.set_yticks(np.arange(0, 101, 5))
        # Add grid lines
        ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.1, color=dark_blue_color, which='both')

        date = str(self.date).replace(" ", "_")
        date = date.replace(":", "-")
        save_path = f"static/model_images/all_models_images/barplot_{date}.png"
        self.barplot_name = f"barplot_{date}.png"
        fig.savefig(save_path, format='png')
        return
    
    
    def bar_plot_for_file(self):
        
        
        model_names = ["File"]
        gold_color = (1.0, 0.84, 0.0)
        dark_green_color = (0.0, 0.0, 0.5)
        dark_green_color = (0.0, 0.5, 0.0)

        # Count the occurrences of each prediction for each model
        

        # Calculate percentages
        percentage_0 = self.file_total_true_percentage
        percentage_1 = self.file_total_fake_percentage

        # Set the figure size and style
        # sns.set(style="whitegrid")  # Use a seaborn style
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotting the grouped bar chart
        width = 0.35  # Width of each bar
        ind = np.arange(len(model_names))

        bars_0 = ax.bar(ind - width/2, percentage_0, width, label='Predicted 0', color=gold_color)
        bars_1 = ax.bar(ind + width/2, percentage_1, width, label='Predicted 1', color=dark_green_color)

        ax.set_xlabel('Model', fontsize=14)
        ax.set_ylabel('Percentage', fontsize=14)
        ax.set_title('Percentage of Predictions for True and Fake by All Models', fontsize=16)
        ax.set_xticks(ind)
        ax.set_xticklabels(model_names, fontsize=12)
        ax.legend(fontsize=12)

        # Add percentage labels on top of each bar with a different color
        for bar_0, bar_1 in zip(bars_0, bars_1):
            height_0 = bar_0.get_height()
            height_1 = bar_1.get_height()
            
            ax.annotate(f'{height_0:.2f}%', xy=(bar_0.get_x() + bar_0.get_width() / 2, height_0),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color='black', fontsize=10)
            
            ax.annotate(f'{height_1:.2f}%', xy=(bar_1.get_x() + bar_1.get_width() / 2, height_1),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color='black', fontsize=10)

        # Set the y-axis scale to show 0-100%
        ax.set_yticks(np.arange(0, 101, 5))
        # Add grid lines
        ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.1, color=dark_green_color, which='both')

        date = str(self.date).replace(" ", "_")
        date = date.replace(":", "-")
        save_path = f"static/model_images/all_models_images/barplot_file_{date}.png"
        self.barplot_file_name = f"barplot_file_{date}.png"
        fig.savefig(save_path, format='png')
        return




if __name__ == "__main__":

    prediction = MultiPrediction()
    print(prediction.get_identity())