FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
# Copy images so the receiver can build its knowledge base
COPY ./KnowledgeBase ./images
CMD ["python", "-u", "app.py"]