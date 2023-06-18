FROM nvcr.io/nvidia/tensorrt:22.10-py3

COPY requirements.txt /workspace

RUN apt-get update && \
    apt install python3.8-venv -y && \
    python -m venv venv && \
    pip install -r requirements.txt