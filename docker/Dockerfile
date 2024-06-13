FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
LABEL org.opencontainers.image.authors="cssherryliang@outlook.com"

WORKDIR /workspace

COPY requirements.txt /workspace

RUN apt update && apt install -y python3-opencv && pip install --no-cache-dir -r requirements.txt
