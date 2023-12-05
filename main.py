from typing import Union
from Sequencial.row_prediction import label_text
from fastapi import FastAPI
import requests
import httpx

app = FastAPI()


@app.get("/")
async def read_root():




    return {"Hello": "World"}


##Ligger och vilar
# @app.post("/post_prediction/")
# async def post_item(input_string: str):
#     api_endpoint = 'http://localhost:8000/predictions/'
#     # headers = {'Content-Type':'application/json'}
#     data = {"text":input_string}
#     print("This is data", data)

#     async with httpx.AsyncClient() as client:
#         response = await client.post(api_endpoint, json=data)
#         print("This is response", response)

#     if response.status_code == 200:
#         result = response.json()
#         return {"prediction": result}
#     else:
#         return{'error',f"Request failed with code {response.status_code}"}
    




@app.post("/predictions/")
async def prediction(input_string: str):
    
    prediction_result = label_text(input_string)

    return {"prediction": prediction_result}







