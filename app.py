from io import BytesIO
from flask import Flask, jsonify, render_template, request, flash, send_file
import requests
from form import textForm, FileForm
import secrets
import os
import os.path
import csv
import pandas as pd

from file_management import check_file_exists, fetch_from_csv, manage_csv_header, get_file_properties, get_first_column_data, is_valid_csv, start_up_process

from multi_model_predict import multi_model_predict_text
from single_model_predict import single_model_predict_text




app = Flask(__name__)
secret_key = secrets.token_hex(32)
app.config['SECRET_KEY'] = secret_key
port = int(os.environ.get('PORT', 5000))





fixed_text = "All vegetarian Sanatan Dharmis only need little care about Social Distancing and enjoy long healthy life."




is_heroku = 'DYNO' in os.environ

if is_heroku:
    current_system = "heroku"
else:
    current_system = "local"




@app.route('/', methods=['GET', 'POST'])
def index():
    
    return render_template("index.html", table=False)




@app.route('/text', methods=['POST', "GET"])
def text():
    check_file_exists()
    form = textForm()
    if request.method == "POST":
        texts = form.text.data
        model_selected = form.model.data

        file_path, new_id = get_file_properties(model_selected, input_type="text")
            
        if model_selected == "all_models":
            
            prediction_object = multi_model_predict_text(texts,input_type="text", new_id=new_id)
            prediction_object.create_histogram()
        else:
            prediction_object = single_model_predict_text(texts, model_selected,input_type="text", new_id=new_id)
        
        

        with open (file_path, 'a') as csvfile:
            
            
            
            if model_selected != "all_models":

                fieldnames = manage_csv_header(file_path, load_from_path=True)
                writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=fieldnames)
                
                

                

                writer.writerow({'id':prediction_object.get_id(), 
                                'text':prediction_object.get_text(),
                                'prediction':prediction_object.get_prediction(),
                                'accuracy':prediction_object.get_accuracy(),
                                'model_selected':prediction_object.get_model_selected(),
                                'date':prediction_object.get_date()})         
            
            else:

                fieldnames = manage_csv_header(file_path, load_from_path=True)
                
                writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=fieldnames)

                
                
                writer.writerow({'id':prediction_object.get_id(),
                                'text':prediction_object.get_text(), 
                                'prediction':prediction_object.get_prediction(), 
                                'accuracy':prediction_object.get_accuracy(),
                                'model_selected':prediction_object.get_model_selected(),
                                'logistic':prediction_object.get_model_vote("logistic"),
                                'SVM':prediction_object.get_model_vote("svm"),
                                'sequential':prediction_object.get_model_vote("sequential"),
                                'date':prediction_object.get_date()})        


        flash('Text submitted and processed successfully!', 'Success!')
        
        return render_template("prediction_result.html",
                                table=True,
                                form=form,
                                prediction_object=prediction_object)


    return render_template("text.html", table=False, form=form)




@app.route('/file', methods=['POST', 'GET'])
def file():
    check_file_exists()
    form = FileForm()
    
    if request.method == "POST":
        model_selected = form.model.data
        csv_file = form.csv_file.data
        file_name = csv_file.filename.split(".")[0]
        
        file_content = csv_file.read()

        file_path, new_group_id = get_file_properties(model_selected, input_type="file")
        
        
        if is_valid_csv(file_content):
            flash('File uploaded and processed successfully!', 'success')
            first_column_data = get_first_column_data(file_content)
            
            if first_column_data is not None:
                
                if model_selected == "all_models":
                    
                    prediction_object = multi_model_predict_text(first_column_data,
                                                          input_type="file",
                                                          file_name=file_name,
                                                          new_group_id=new_group_id)
                else:
                    prediction_object = single_model_predict_text(first_column_data,
                                                        model_selected,
                                                        file_name=file_name,
                                                        new_group_id=new_group_id)
                
                

                with open (file_path, 'a', encoding='utf-8') as csvfile:
                    fieldnames=manage_csv_header(file_path, load_from_path=True)
                    writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=fieldnames)
                    
                    if model_selected != "all_models":
                    
                        for object in prediction_object.get_all_prediction_objects():
                            writer.writerow({
                                            'group_id':prediction_object.get_group_id(),
                                            'prediction_id':object.get_id(),
                                            'text':object.get_text(),
                                            'prediction':object.get_prediction(),
                                            'accuracy':object.get_accuracy(),
                                            'model_selected':object.get_model_selected(),
                                            'date':object.get_date()})            
                    else:
                        for object in prediction_object.get_all_prediction_objects():
                            writer.writerow({
                                            'group_id':prediction_object.get_group_id(),
                                            'prediction_id':object.get_id(),
                                            'text':object.get_text(),
                                            'prediction':object.get_prediction(),
                                            'accuracy':object.get_accuracy(),
                                            'model_selected':object.get_model_selected(),
                                            'logistic':object.get_model_vote("logistic"),
                                            'SVM':object.get_model_vote("svm"),
                                            'sequential':object.get_model_vote("sequential"),
                                            'date':object.get_date()})
                
                
                
                return render_template("prediction_result.html",
                                       table=True,
                                       form=form,
                                       prediction_object=prediction_object)

            else:
                flash('First column data in CSV file must contain rows of text', 'error')
        else:
            flash('Error: The first column of the CSV file must contain text.', 'error')

    else:
        form_errors = form.errors
        return render_template("file.html", table=False, form=form, form_errors=form_errors)
    
    return render_template("file.html", table=False, form=form)



######## API ########
    


@app.route("/<path:url>")
def barplot(url):
    try:

        response = requests.get(url)
        response.raise_for_status()
        
        image_data = BytesIO(response.content)

        return send_file(image_data, mimetype='image')

    except requests.exceptions.RequestException as e:
        return f"Error fetching image: {e}", 500
    


@app.route("/file_api/<model_selected>/<id>")
def more_predictions(id, model_selected):
    
    id = int(id)
    prediction_list = []
    page = int(request.args.get('page', 1))
    
    
    if model_selected != "multi":
        current_predictions = fetch_from_csv("stored_data/file_predictions_single_model.csv", id)
        
        
        predictions = current_predictions[(page-1)*10:page*10]
        
        for prediction in predictions:
            t = {
                "groupId":id,
                "Id": prediction.get_id(),
                "Text": prediction.get_text(),
                "Prediction": prediction.get_prediction(),
                "Accuracy": prediction.get_accuracy(),
                "ModelSelected": prediction.get_model_selected(),
                "Date": prediction.get_date()
            }
            prediction_list.append(t)
        
        return jsonify(prediction_list)
    
    else:
        current_predictions = fetch_from_csv("stored_data/file_predictions_multi_model.csv", id)
        predictions = current_predictions[(page-1)*10:page*10]
        
        for prediction in predictions:
            t = {
                "groupId":id,
                "Id": prediction.get_id(),
                "Text": prediction.get_text(),
                "Prediction": prediction.get_prediction(),
                "Accuracy": prediction.get_accuracy(),
                "ModelSelected": prediction.get_model_selected(),
                "Logistic": prediction.get_model_vote("logistic"),
                "SVM": prediction.get_model_vote("svm"),
                "Sequential": prediction.get_model_vote("sequential"),
                "Date": prediction.get_date()
            }
            prediction_list.append(t)
        
        return jsonify(prediction_list)





@app.route('/api/<model>/predict', methods=['POST'])
def sequential_predict_text(model):
    
    models = ["logistic","svm","sequential","all_models"]
    if model in models:
        if request.method == 'POST':
            
            data = request.get_json()
            input_text = data.get('text', "")
            if model == "all_models":
                prediction_object = multi_model_predict_text(input_text, input_type="api")
                return jsonify({"multi_prediction": {
                                    
                                    'id':prediction_object.get_id(),
                                    'text':prediction_object.get_text(), 
                                    'prediction':prediction_object.get_prediction(), 
                                    'accuracy':prediction_object.get_accuracy(),
                                    'model_selected':prediction_object.get_model_selected(),
                                    'logistic':prediction_object.get_model_vote("logistic"),
                                    'SVM':prediction_object.get_model_vote("svm"),
                                    'sequential':prediction_object.get_model_vote("sequential"),
                                    'date':prediction_object.get_date()}
                                
                                })
                                
            else:
                
                prediction_object = single_model_predict_text(input_text, model, input_type="api")
                
                
                return jsonify({"single_prediction": {
                                                        "id": prediction_object.get_id(),
                                                        "text": prediction_object.get_text(),
                                                        "prediction": prediction_object.get_prediction(),
                                                        "accuracy": prediction_object.get_accuracy(),
                                                        "model_selected": prediction_object.get_model_selected(),
                                                        "date": prediction_object.get_date()
                                                        }
                                })     
        else:
            return print("POST request required")
    else:
        return jsonify({"error": "Model not found. Available models: logistic, svm, sequential, all_models"})



if __name__ == "__main__":
    start_up_process(current_system)
    
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader = False)

    
