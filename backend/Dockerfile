FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt .
RUN pip install --requirement requirements.txt --target "${LAMBDA_TASK_ROOT}" --upgrade --no-cache-dir

COPY main.py ${LAMBDA_TASK_ROOT}
COPY database.py ${LAMBDA_TASK_ROOT}
COPY api ${LAMBDA_TASK_ROOT}/api

CMD [ "main.handler" ]
