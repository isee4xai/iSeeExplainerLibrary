FROM python:3.8-slim-buster

WORKDIR /app
COPY requirements.txt requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD [ "python", "app"]
