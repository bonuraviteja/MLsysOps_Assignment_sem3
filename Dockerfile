FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
COPY artifacts ./artifacts

RUN pip install --no-cache-dir -e .

ENV MODEL_PATH=/app/artifacts/model/model.joblib
EXPOSE 8000

CMD ["uvicorn", "heart_disease_mlops.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
