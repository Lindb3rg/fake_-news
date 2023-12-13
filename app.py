from flask import Flask, jsonify, render_template, request
# from Sequencial.row_prediction import label_text
from SVM.row_prediction_svm import svm_label_text
import os

app = Flask(__name__)
port = int(os.environ.get('PORT', 5000))




# Invoke-RestMethod -Uri 'http://127.0.0.1:5000/api/sequential/predict' -Method Post -Headers @{"Content-Type"="application/json"} -Body '{"text": "All vegetarian Sanatan Dharmis only need little care about Social Distancing and enjoy long healthy life."}'
# Invoke-RestMethod -Uri 'http://127.0.0.1:5000/api/svm/predict' -Method Post -Headers @{"Content-Type"="application/json"} -Body '{"text": "All vegetarian Sanatan Dharmis only need little care about Social Distancing and enjoy long healthy life."}'







fixed_text = "All vegetarian Sanatan Dharmis only need little care about Social Distancing and enjoy long healthy life."

# @app.route('/api/sequential/predict', methods=['GET', 'POST'])
# def sequential_predict_text():
#     if request.method == 'POST':
#         # Get the text from the POST request
#         data = request.get_json()
#         input_text = data.get('text', "")

#         # Use the fixed text for preprocessing (replace this with your actual preprocessing logic)
#         predict_text = label_text(input_text)

#         # Return the preprocessed text as JSON response for POST requests
#         response_data = {'predict': predict_text}
#         return jsonify(response_data)
#     else:
#         # Render an HTML page for GET requests
#         predict_text = label_text(fixed_text)
#         return jsonify({"Original Text": fixed_text, "Sequential Prediction Result": predict_text})
    

@app.route('/api/svm/predict', methods=['GET', 'POST'])
def svm_predict_text():
    if request.method == 'POST':
        # Get the text from the POST request
        data = request.get_json()
        input_text = data.get('text', "")

        # Use the fixed text for preprocessing (replace this with your actual preprocessing logic)
        predict_text = svm_label_text(input_text)

        # Return the preprocessed text as JSON response for POST requests
        response_data = {'predict': predict_text}
        return jsonify(response_data)
    else:
        # Render an HTML page for GET requests
        predict_text = svm_label_text(fixed_text)
        return jsonify({"Original Text": fixed_text, "SVM Prediction Result": predict_text})
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, debug=True)

    
