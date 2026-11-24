FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app

COPY trainr/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ common/
COPY trainr/ .

# run as a non-root user
RUN useradd --create-home appuser && mkdir -p models && chown appuser models
USER appuser

EXPOSE 7777
CMD ["python", "main.py"]
