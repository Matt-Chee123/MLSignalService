FROM python:3.12.4-slim
WORKDIR /app

COPY data_pipeline/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY data_pipeline/ .

RUN useradd app
USER app

CMD ["python", "data_orchestrator.py"]