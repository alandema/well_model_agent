FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY init_db.py ./
COPY entrypoint.sh ./

RUN chmod +x entrypoint.sh

EXPOSE 8601

CMD ["./entrypoint.sh"]