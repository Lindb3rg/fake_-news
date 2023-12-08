import requests as rq
from datetime import datetime


BASE_URL = 'https://lindb3rg.pythonanywhere.com'

text = "The Chairman of the Republican Party of Texas said the recovery rate for COVID-19 is 99.9% in Texas. That’s False. @PolitiFactTexas https://t.co/GmUXoVT2Dh https://t.co/ltIHWPchJM"
text_1 = "Tjoflöt"
payload = {"input":text}

response = rq.get(BASE_URL, params=payload)

json_values = response.json()

rq_input = json_values["input"]
prediction_result = json_values['prediction_result']
timestamp = json_values['timestamp']

print(f"Input is: {rq_input}")
print(f"prediction_result is: {prediction_result}")
print(f"Timestamp is: {datetime.fromtimestamp(timestamp)}")