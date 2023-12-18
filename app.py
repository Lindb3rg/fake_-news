from flask import Flask, jsonify, render_template, request
from model_predict import model_predict_text
import os
import csv


app = Flask(__name__)
port = int(os.environ.get('PORT', 5000))


# Invoke-RestMethod -Uri 'http://127.0.0.1:5000/api/sequential/predict' -Method Post -Headers @{"Content-Type"="application/json"} -Body '{"text": "All vegetarian Sanatan Dharmis only need little care about Social Distancing and enjoy long healthy life."}'
# Invoke-RestMethod -Uri 'http://127.0.0.1:5000/api/svm/predict' -Method Post -Headers @{"Content-Type"="application/json"} -Body '{"text": "All vegetarian Sanatan Dharmis only need little care about Social Distancing and enjoy long healthy life."}'



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

        first_column_data = [row[0] for row in csv_reader]

        return first_column_data
    except csv.Error:
        return None


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html", table=False)












































@app.route("/prediction", methods=["POST"])
def prediction_text():
    if request.method == "POST":
        model_selected = request.form["model"]
        texts = request.form.get("textArea")
        texts = texts.split("\n")

        uploaded_file = request.files.get('fileInput')
        if uploaded_file:
            file_content = uploaded_file.read()
            
            if not is_valid_csv(file_content):
                error = "Uploaded file is not a valid CSV file."
                return render_template("error_page.html", error=error)
            
            first_column_data = get_first_column_data(file_content)
            if first_column_data is not None:
                texts.extend(first_column_data)
            else:
                error = "First column data in csv file must contain rows of text"
                return render_template("error_page.html", error=error)

      
        
        
        predictions = model_predict_text(texts, model_selected)


        return render_template("index.html", predictions=predictions, table=True, model_selected=model_selected)
    



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

    
