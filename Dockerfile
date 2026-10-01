FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN python -m unittest -v test_service_auth test_security_contract

RUN apt-get update && apt-get install -y build-essential libssl-dev \
  && pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
