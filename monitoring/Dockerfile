FROM python:3.12

WORKDIR /app

COPY monitoring/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY monitoring/ ./monitoring/

CMD ["python", "-m", "analysis.orchestrator"]