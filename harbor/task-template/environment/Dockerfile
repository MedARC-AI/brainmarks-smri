FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy==2.5.4 \
    pandas==3.1.0 \
    scipy==1.18.1 \
    scikit-learn==1.9.1 \
    scikit-image==0.26.0 \
    nibabel==5.4.2 \
    nilearn==0.14.1 \
    matplotlib==3.11.2

ENV MPLBACKEND=Agg

RUN useradd --create-home agent && mkdir /output && chown agent:agent /output

COPY data /data

WORKDIR /home/agent
