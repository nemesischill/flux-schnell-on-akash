FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS build
ENV USER=nemesischill
ENV APP=app
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y build-essential ffmpeg && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && useradd -m $USER
FROM build AS runtime
RUN mkdir -p /opt/$APP && chown $USER:$USER /opt/$APP
WORKDIR /opt/$APP
RUN pip install poetry
COPY --chown=$USER:$USER pyproject.toml pyproject.toml
RUN poetry config virtualenvs.create false && poetry install --with=dev --no-root --no-interaction && rm -rf /root/.cache/pypoetry
USER $USER
COPY --chown=$USER:$USER app.py app.py
COPY --chown=$USER:$USER settings.yaml settings.yaml
ENTRYPOINT ["python", "app.py"]
