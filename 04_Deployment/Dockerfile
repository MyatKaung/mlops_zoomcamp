FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Install required packages
RUN pip install pandas scikit-learn pyarrow requests

# Copy the scoring script
COPY score_docker.py ./

# Set the entrypoint
ENTRYPOINT ["python", "score_docker.py"]