FROM python:3.8-slim-buster as base

FROM base AS builder
RUN apt-get -y update  && apt-get -y install git
COPY requirements.txt /tmp/requirements.txt
RUN pip install --target=/tmp/build --upgrade pip
RUN pip install --target=/tmp/build -r /tmp/requirements.txt

FROM base as runtime

WORKDIR /app
COPY --from=builder /tmp/build/ /usr/local/lib/python3.8/site-packages/
COPY . .

CMD [ "python", "app.py"]
