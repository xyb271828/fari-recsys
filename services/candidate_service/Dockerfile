FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.3.5 faiss-cpu==1.11.0 fastapi==0.115.12 uvicorn==0.34.2
COPY services /app/services
ENV FARI_ARTIFACT_DIR=/artifacts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EXPOSE 8000
CMD ["uvicorn", "services.candidate_service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
