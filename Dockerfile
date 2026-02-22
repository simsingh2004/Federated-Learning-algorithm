FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install --no-cache-dir flwr torchmetrics tqdm flwr-datasets

COPY . /app