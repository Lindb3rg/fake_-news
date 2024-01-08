from flask import Flask, jsonify, render_template, request, flash
from model_predict import model_predict_text
from form import textForm, FileForm
import secrets
import os
import csv


app = Flask(__name__)
secret_key = secrets.token_hex(32)
app.config['SECRET_KEY'] = secret_key
port = int(os.environ.get('PORT', 5000))





fixed_text = "All vegetarian Sanatan Dharmis only need little care about Social Distancing and enjoy long healthy life."

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






@app.route('/', methods=['GET', 'POST'])
def index():
    
    return render_template("index.html", table=False)




@app.route('/text', methods=['POST', "GET"])
def text():
    form = textForm()
    if request.method == "POST":
        texts = form.text.data
        model_selected = form.model.data

        predictions, row_predictions = model_predict_text(texts, model_selected)
        flash('Text submitted and processed successfully!', 'Success!')
        return render_template("prediction_result.html", predictions=predictions, table=True, row_predictions=row_predictions, model_selected=model_selected, form=form)


    return render_template("text.html", table=False, form=form)





@app.route('/file', methods=['POST', 'GET'])
def file():
    form = FileForm()
    
    if request.method == "POST":
        model_selected = form.model.data
        css_file = form.css_file.data
        file_content = css_file.read()
        
        if is_valid_csv(file_content):
            flash('File uploaded and processed successfully!', 'success')
            first_column_data = get_first_column_data(file_content)
            
            if first_column_data is not None:
                predictions, row_predictions = model_predict_text(first_column_data, model_selected)
                return render_template("prediction_result.html", predictions=predictions, row_predictions=row_predictions, table=True, model_selected=model_selected, form=form)

            else:
                flash('First column data in CSV file must contain rows of text', 'error')
        else:
            flash('Error: The first column of the CSV file must contain text.', 'error')

    else:
        form_errors = form.errors
        return render_template("file.html", table=False, form=form, form_errors=form_errors)
    
    return render_template("file.html", table=False, form=form)




    



@app.route('/api/sequential/predict', methods=['GET', 'POST'])
def sequential_predict_text():

    if request.method == 'POST':
        
        data = request.get_json()
        input_text = data.get('text', "")
        
        
        predict_text = model_predict_text(input_text, "sequential")
        return jsonify(predict_text)
    else:

        
        predict_text = model_predict_text(fixed_text, "sequential")
        return jsonify({"Original Text": fixed_text, "Sequential Prediction Result": predict_text})


@app.route('/api/svm/predict', methods=['GET', 'POST'])
def svm_predict_text():
    if request.method == 'POST':
        
        data = request.get_json()
        input_text = data.get('text', "")
        predict_text = model_predict_text(input_text, "svm")
        return jsonify(predict_text)
    
    else:
        
        predict_text = model_predict_text(fixed_text, "svm")
        return jsonify({"Original Text": fixed_text, "SVM Results": predict_text})


@app.route('/api/logistic/predict', methods=['GET', 'POST'])
def logistic_predict_text():
    if request.method == 'POST':

        data = request.get_json()
        input_text = data.get('text', "")        
        predict_text = model_predict_text(input_text, "logistic")

        return jsonify(predict_text)
    else:
        
        predict_text = model_predict_text(fixed_text, "logistic")
        return jsonify({"Original Text": fixed_text, "Logistic Model": predict_text})


@app.route('/api/all_models/predict', methods=['GET', 'POST'])
def all_models_predict_text():
    if request.method == 'POST':
        
        data = request.get_json()
        input_text = data.get('text', "")        
        predict_text = model_predict_text(input_text, "all_models")

        return jsonify(predict_text)
    else:
        predict_text = model_predict_text(fixed_text, "all_models")
        return jsonify({"Original Text": fixed_text, "Best result från all Models": predict_text})
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader = False)

    
