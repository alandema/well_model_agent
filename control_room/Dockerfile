FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8601

# main_write appends sensor readings to the database while main_read serves
# the control-room dashboard on the same container.
CMD ["sh", "-c", "python app/main_write.py & streamlit run app/main_read.py --server.port 8601 --server.address 0.0.0.0"]
