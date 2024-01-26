FROM jupyter/scipy-notebook


WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt



ENTRYPOINT [ "jupyter-lab", "--NotebookApp.password=''", "--NotebookApp.token=''" ]