FROM python:3.8-slim-buster as base

FROM base AS builder
RUN apt-get -y update  && apt-get -y install git
COPY requirements.txt /tmp/requirements.txt
RUN pip install --target=/tmp/build --upgrade pip
RUN pip install --target=/tmp/build -r /tmp/requirements.txt

FROM base as runtime

WORKDIR /app
RUN apt update && apt-get install -y libgl1  libglib2.0-0  chromium

# CHROMIUM default flags for container environnement
# The --no-sandbox flag is needed by default since we execute chromium in a root environnement
RUN echo 'export CHROMIUM_FLAGS="$CHROMIUM_FLAGS --no-sandbox"' >> /etc/chromium.d/default-flags


COPY --from=builder /tmp/build/ /usr/local/lib/python3.8/site-packages/
COPY . .

CMD python app.py ${UPLOAD_FOLDER}
