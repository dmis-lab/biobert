FROM tensorflow/tensorflow:1.15.5-gpu-py3

RUN apt-get update -y && apt-get install -y python3-dev build-essential wget

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

RUN ./download.sh

ENV BIOBERT_DIR=./biobert_v1.1_pubmed

ENV NER_DIR=./datasets/NER/NCBI-disease

ENV OUTPUT_DIR=./ner_outputs

RUN mkdir -p $OUTPUT_DIR