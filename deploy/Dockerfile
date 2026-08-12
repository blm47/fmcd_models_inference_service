# GPU-образ для H100: CUDA runtime + torch с GPU-поддержкой
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3.11 python3-pip

WORKDIR /srv/app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Артефакты модели вшиты в образ на v1
COPY artifacts/ ./artifacts/
COPY configs/ ./configs/
COPY fmcd/ ./fmcd/
COPY app/ ./app/

EXPOSE 8000

# Один воркер-процесс: конкурентность ограничена одной активной GPU-задачей
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
