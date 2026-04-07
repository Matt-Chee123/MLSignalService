FROM python:3.12

WORKDIR /app

COPY backtest/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backtest/ ./backtest/
COPY data_pipeline/ ./data_pipeline/
COPY config/ ./config/

CMD ["python", "-m", "backtest.backtest_orchestrator"]
