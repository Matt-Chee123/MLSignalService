FROM public.ecr.aws/lambda/python:3.12

COPY monitoring/requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt -t ${LAMBDA_TASK_ROOT}

COPY monitoring/ ${LAMBDA_TASK_ROOT}/monitoring/

CMD ["monitoring.orchestrator.lambda_handler"]