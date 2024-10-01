FROM python:3.11-bookworm
LABEL authors="Arturo Ortiz"

WORKDIR /app

EXPOSE 8501

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENTRYPOINT ["fastapi", "dev", "main.py", "--port", "8000"]