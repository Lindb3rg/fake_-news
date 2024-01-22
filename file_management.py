import csv
import os
import pandas as pd


def path_decomposer(path:str)->tuple:
    path = path.replace("stored_data/", "")
    path = path.replace(".csv", "")
    input_type,_,models,_ = path.split("_")
    return (input_type, models)






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
            with open(csv_file_path, 'w') as csvfile:
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
        df = pd.read_csv(csv_file_path)
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
    df = pd.read_csv(csv_file)
    df = df.dropna(how='all')
    df.to_csv(csv_file, index=False)


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
            with open(file_path, 'r', newline='') as csvfile:
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