# Note that this container must be built from "backend" directory.
FROM hastyai/inference:latest

ADD service/Pipfile .
ADD service/Pipfile.lock .
RUN pip install pipenv \
    && pipenv install --system --deploy


COPY model /inference/model
COPY service /inference/service

WORKDIR /inference
COPY service/main.py .


ENTRYPOINT ["python3", "main.py"]
