FROM python:3.6
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY RandomForestClassifier.pkl /RandomForestClassifier.pkl
COPY app.py /app.py
COPY classes.py /classes.py
COPY another_train.yaml /another_train.yaml

WORKDIR .

ENV PATH_TO_MODEL="/RandomForestClassifier.pkl"
ENV PATH_TO_DESC="/another_train.yaml"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
