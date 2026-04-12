FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip

WORKDIR /app

COPY training/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY training/ ./training/
COPY models/ ./models/
COPY config/ ./config/

CMD ["python3", "-m", "training.orchestrator"]