FROM python:3.12-slim

WORKDIR /app

COPY inference/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY inference/ ./inference/
COPY data_pipeline/ ./data_pipeline/
COPY config/ ./config/

EXPOSE 8000

CMD ["uvicorn", "inference.main:app", "--host", "0.0.0.0", "--port", "8000"]
