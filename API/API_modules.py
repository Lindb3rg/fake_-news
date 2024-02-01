import requests as rq

"""
This function is used to POST a text to one or all of our models. Connect automation and database to own choosing. 

Post query can also be done via curl, example:

curl -X POST -H "Content-Type: application/json" -d '{"text":<text_for_prediction>}' -k http://<web adress>/api/<model of choosing>/predict

current_models = [
    
    "svm",
    "sequential",
    "logistic",
    "all_models"]

"""






def post_to_our_API(text:str, model:str)->dict:

    
        
    BASE_URL = f'https://<web adress>/api/{model}/predict'
    payload = {"text":text}
    
    response = rq.post(BASE_URL, json=payload)


    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.text)
    





    