FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UI_SERVER_HOST=0.0.0.0 \
    UI_SERVER_PORT=8080

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY catalog/ catalog/
COPY examples/ examples/
COPY scenarios/ scenarios/
COPY src/ src/
COPY tests/ tests/
COPY ui.html .
COPY ui_server.py .
COPY run_all.py .
COPY README.md .
COPY TELEPITES_UTMUTATO.md .

RUN mkdir -p /app/output/results /app/output/generated_scenarios

EXPOSE 8080

CMD ["python", "ui_server.py"]
