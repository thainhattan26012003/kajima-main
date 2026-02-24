# Builder
FROM public.ecr.aws/lambda/python:3.10 AS builder

RUN pip install poetry==2.0.1 --no-cache-dir

ENV POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_IN_PROJECT=1 \
  POETRY_VIRTUALENVS_CREATE=1 \
  POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /code

COPY pyproject.toml poetry.lock ./

RUN poetry self add poetry-plugin-export
RUN poetry export \
  --with=batch_processor \
  --without=dev,notebook \
  --format requirements.txt \
  --output requirements.txt && \
  pip install --no-cache-dir --target ./package -r requirements.txt

# Runner
FROM public.ecr.aws/lambda/python:3.10 AS runner

ENV VIRTUAL_ENV=/code/.venv \
  PATH=/code/.venv/bin:$PATH \
  HF_HOME=/tmp/.cache/huggingface

WORKDIR ${LAMBDA_TASK_ROOT}

COPY --from=builder /code/package ${LAMBDA_TASK_ROOT}

# Copy all project files - assuming models in checkpoints directory
COPY ./src ${LAMBDA_TASK_ROOT}/src
COPY ./checkpoints ${LAMBDA_TASK_ROOT}/checkpoints
COPY ./config ${LAMBDA_TASK_ROOT}/config
COPY ./libs ${LAMBDA_TASK_ROOT}/libs

# Ensure Hugging Face cache directory exists
RUN mkdir -p /tmp/.cache/huggingface

ENV PYTHONBUFFERED=1
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}:${PYTHONPATH}

# Lambda handler in format: module.handler
CMD [ "src.batch_processor.lambda_handler" ]