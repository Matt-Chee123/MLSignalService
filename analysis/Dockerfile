FROM python:3.12

WORKDIR /app

COPY analysis/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY analysis/ ./analysis/
COPY config/ ./config/

CMD ["python", "-m", "analysis.orchestrator"]