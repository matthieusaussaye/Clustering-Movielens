FROM python:3.8.3-slim
RUN mkdir -p /app \
  && mkdir -p /app/data/
WORKDIR /app
RUN pip install poetry
COPY pyproject.toml ./
RUN poetry install








