# Simple Dockerfile for Phala Cloud TEE Deployment
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install flask

COPY basic_example.py .

ENV PORT=4768

EXPOSE 4768

CMD ["python", "basic_example.py"]
