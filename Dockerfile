FROM python:3.8-slim-buster as base

FROM base AS builder
COPY requirements.txt /tmp/requirements.txt
RUN pip install --target=/tmp/build --upgrade
RUN pip install --target=/tmp/build -r /tmprequirements.txt

FROM base as runtime

WORKDIR /app
COPY --from=builder /tmp/build/ /usr/local/lib/python3.8/site-packages/
COPY . .

CMD [ "python", "app.py"]
