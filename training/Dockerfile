FROM python:3.12

WORKDIR /app

COPY training/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY training/ ./training/
COPY models/ ./models/
COPY config/ ./config/

CMD ["python", "-m", "training.orchestrator"]