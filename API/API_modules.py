import requests as rq
from flask_wtf import csrf

"""
This function is used to POST a text to one or all of our models. 

"""

# def get_csrf_token():
    

#     csrf_response = rq.get('https://fake-news-version-1-8664e27edd64.herokuapp.com/get_csrf_token')

#     if csrf_response.status_code == 200:
#         return csrf_response.text
#     else:
#         return None


def post_to_our_API(text:str, model:str)->dict:

    current_models = ["svm",
                      "sequential",
                      "logistic",
                      "all_models"]
    
    if model in current_models:
        
        BASE_URL = f'https://fake-news-version-1-8664e27edd64.herokuapp.com/api/{model}/predict'
        payload = {"text":text}
        
        response = rq.post(BASE_URL, json=payload)


        if response.status_code == 200:
            return response.json()
        else:
            print("Error:", response.text)
    else:
        return f"Model not found"





    