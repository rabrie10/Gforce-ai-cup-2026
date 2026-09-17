FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY agent_server.py .

EXPOSE 9052

CMD ["python", "agent_server.py"]