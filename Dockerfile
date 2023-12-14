FROM python:3.11.4
WORKDIR /app
COPY . /app/
COPY ./requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
CMD python ./app.py


