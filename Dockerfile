# Dockerfile
FROM python:3.11-slim

# évite buffering et permet logs en temps réel
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# copier requirements en premier pour utiliser cache docker
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --upgrade pip \
    && pip install -r /app/requirements.txt \
    && apt-get remove -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# copier le reste
COPY . /app

# créer dossier plots si nécessaire
RUN mkdir -p /app/plots

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
