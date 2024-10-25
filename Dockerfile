FROM python:3.11-bookworm
LABEL authors="Arturo Ortiz"

WORKDIR /app



RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000

ENTRYPOINT ["fastapi", "run", "main.py", "--port", "80"]