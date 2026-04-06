FROM python:3.12

WORKDIR /app

COPY data_pipeline/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY data_pipeline/ ./data_pipeline/
COPY config/ ./config/

CMD ["python", "-m", "data_pipeline.data_orchestrator"]