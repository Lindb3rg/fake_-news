import csv
import os
import pandas as pd
from model import SinglePrediction, MultiPrediction
import platform
import shutil

# Get the system's operating system
os_system = platform.system()




    




def path_decomposer(path:str)->tuple:
    path = path.replace("stored_data/", "")
    path = path.replace(".csv", "")
    input_type,_,models,_ = path.split("_")
    return (input_type, models)



def fetch_from_csv(csv_file_path, group_id):
    input_type = path_decomposer(csv_file_path)
    df = pd.read_csv(csv_file_path, encoding='utf-8')
    group_id = int(group_id)
    group_rows = df[df['group_id'] == group_id]

    predictions = []
    if input_type[1] == "single":
        for _, row in group_rows.iterrows():
            
            single_object = SinglePrediction()
            single_object.id=row['prediction_id']
            single_object.text=row['text']
            single_object.prediction=row['prediction']
            single_object.accuracy=row['accuracy']
            single_object.model_selected=row['model_selected']
            single_object.date=row['date']

            predictions.append(single_object)
    elif input_type[1] == "multi":
        for _, row in group_rows.iterrows():
            multi_object = MultiPrediction()
            multi_object.id = row['prediction_id']
            multi_object.text = row['text']
            multi_object.prediction = row['prediction']
            multi_object.accuracy = row['accuracy']
            multi_object.model_selected = row['model_selected']
            multi_object.set_model_vote("logistic", row['logistic'])
            multi_object.set_model_vote("svm", row['SVM'])
            multi_object.set_model_vote("sequential", row['sequential'])
            multi_object.date = row['date']
            predictions.append(multi_object)

    return predictions



def clear_folder(folder_path, limit):
    
    files = os.listdir(folder_path)

    
    if len(files) > limit:
        print(f"Number of files ({len(files)}) exceeds the limit ({limit}). Clearing the folder...")
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Error: {e}")

        print("Folder cleared.")
    else:
        print("Number of files within the limit. No action required.")


def clear_csv_file(csv_path, limit):
    
    csv = pd.read_csv(csv_path)
    

    
    if len(csv) > limit:
        
        print(f"Number of rows ({len(csv)}) exceeds the limit ({limit}). Clearing the folder...")
        csv = pd.DataFrame(columns=csv.columns)
        csv.to_csv(csv_path, index=False)
        print("Folder cleared.")
        
    else:
        print("Number of files within the limit. No action required.")






def manage_csv_header(csv_file_path:str, create_from_path: bool=False,load_from_path:bool=False)->None|list:
    
    header_text_single = ['id','text','prediction','accuracy','model_selected', 'date']    
    header_text_multi = ['id','text','prediction','accuracy','model_selected','logistic', 'SVM','sequential','date']
    header_file_single = ['group_id','prediction_id','text','prediction','accuracy','model_selected', 'date']
    header_file_multi = ['group_id','prediction_id','text','prediction','accuracy','model_selected','logistic', 'SVM','sequential','date']
    
    input_type_tuple = path_decomposer(csv_file_path)
    
    if load_from_path:
        if input_type_tuple == ('text', 'single'):
            return header_text_single
        
        elif input_type_tuple == ('text', 'multi'):
            return header_text_multi
        
        elif input_type_tuple == ('file', 'single'):
            return header_file_single
        
        elif input_type_tuple == ('file', 'multi'):
            return header_file_multi
    
    elif create_from_path:
        try:
            with open(csv_file_path, 'w', encoding='utf-8', errors='replace') as csvfile:
                if input_type_tuple == ('text', 'single'):

                    fieldnames = header_text_single

                elif input_type_tuple == ('text', 'multi'):
                    fieldnames = header_text_multi

                elif input_type_tuple == ('file', 'single'):
                    fieldnames = header_file_single

                elif input_type_tuple == ('file', 'multi'):
                    fieldnames = header_file_multi
                    

                writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=fieldnames)
                writer.writeheader()
                return
            
        except Exception as e:
            print(f"Error: {e}")
            return None


def fetch_last_id_from_csv(csv_file_path:str,input_type:str)-> int:
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        if not df.empty:
            if input_type == "file":
                
                latest_id = df['group_id'].max()
                return latest_id + 1

            elif input_type == "text":
                latest_id = df['id'].max()
                return latest_id + 1
        else:
            
            return 1
    
    except Exception as e:
        
        print(f"Error: {e}")
        return None




def get_file_properties(model_selected, input_type):

    
    if input_type == "text":
        
        
        if model_selected != "all_models":
            path = "stored_data/text_predictions_single_model.csv"
            
        else:
            path = "stored_data/text_predictions_multi_model.csv"
        
        
    elif input_type == "file":

        if model_selected != "all_models":
            path = "stored_data/file_predictions_single_model.csv"
            
        else:
            path = "stored_data/file_predictions_multi_model.csv"
            
            
    id = fetch_last_id_from_csv(path,input_type)    
        

    return path, id



def remove_empty_rows(csv_file):
    df = pd.read_csv(csv_file, encoding='utf-8')
    df = df.dropna(how='all')
    df.to_csv(csv_file, index=False, encoding='utf-8')


def check_file_exists():
    list_of_files = [
        "stored_data/text_predictions_single_model.csv",
        "stored_data/text_predictions_multi_model.csv",
        "stored_data/file_predictions_single_model.csv",
        "stored_data/file_predictions_multi_model.csv"
    ]
    for file_path in list_of_files:
        file_exists = os.path.isfile(file_path)
        if not file_exists:
            manage_csv_header(file_path, create_from_path=True)
        
        else:
            with open(file_path, 'r', newline='', encoding='utf-8', errors='replace') as csvfile:
                reader = csv.reader(csvfile)
        
                
                try:
                    first_row = next(reader)
                    if any(field.isalnum() for field in first_row):
                        remove_empty_rows(file_path)
                    else:
                        manage_csv_header(file_path, create_from_path=True)
                        
                except StopIteration:
                    
                    manage_csv_header(file_path, create_from_path=True)
                
                
                
                    
        
      

                    
        
        
    
        
        
        
   



def is_valid_csv(file_content):
    try:
        import csv
        csv.reader(file_content.decode('utf-8'))
        return True
    except csv.Error:
        return False
    
def get_first_column_data(file_content):
    try:
        decoded_content = file_content.decode('utf-8')
        csv_reader = csv.reader(decoded_content.splitlines())
        next(csv_reader)
        first_column_data = [row[0] for row in csv_reader]

        return first_column_data
    except csv.Error:
        return None