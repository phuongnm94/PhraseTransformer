FROM python:3.7-slim-bullseye
COPY . /workspace
COPY ../src/ /workspace/src

RUN pip install -r /workspace/requirements.txt

WORKDIR /workspace
CMD ["python", "service/app.py"]
