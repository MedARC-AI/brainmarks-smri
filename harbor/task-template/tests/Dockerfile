FROM python:3.13-slim

RUN pip install --no-cache-dir \
    numpy==2.5.4 \
    pandas==3.1.0 \
    scikit-learn==1.9.1 \
    nibabel==5.4.2

COPY . /tests/
