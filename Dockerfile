FROM continuumio/miniconda3

RUN mkdir -p /app
WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "dspipeline-detectron2-licenseplates", "/bin/bash", "-c"]

RUN mkdir -p /app/assests

# Copy the necesary parts --> replace with src in future
COPY output .
COPY config .
COPY assests .
COPY transformers .
COPY app.py .
COPY licenseplates_detector.py .
COPY router .


EXPOSE 8888

CMD ["conda", "run", "--no-capture-output", "-n", "dspipeline-detectron2-licenseplates","uvicorn", "--host", "0.0.0.0", "--port", "8888", "api-licenseplates.app:app"]


# CMD ["uvicorn", "--host", "0.0.0.0", "--port", "5000", "api-licenseplates.app:app"]
