FROM public.ecr.aws/docker/library/python:3.13.4-slim AS installer-image

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update -y 2>/dev/null >/dev/null && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq install -y \
    build-essential \
    curl \
    software-properties-common 2>/dev/null >/dev/null \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --user --no-cache-dir --upgrade -q -q pip && \
    pip install --user --no-cache-dir -q -q -r /app/requirements.txt


FROM public.ecr.aws/docker/library/python:3.13.4-slim
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update -y 2>/dev/null >/dev/null && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq upgrade -y 2>/dev/null >/dev/null && \
    DEBIAN_FRONTEND=noninteractive apt-get install -qq -y \
    curl \
    ca-certificates \
    bash \
    git \
    openssh-client \
    kubernetes-client 2>/dev/null >/dev/null && \
    addgroup --gid 8000 fastapi && \
    adduser --uid 8000 --gid 8000 --disabled-password --gecos "" fastapi
USER fastapi
WORKDIR /home/fastapi
COPY --chown=fastapi:fastapi --from=installer-image /root/.local /home/fastapi/.local/
COPY --chown=fastapi:fastapi app/ /home/fastapi/app/
RUN chown -R fastapi:fastapi /home/fastapi/app/
COPY --chown=fastapi:fastapi tools/ /home/fastapi/tools/
RUN chown -R fastapi:fastapi /home/fastapi/tools/
COPY --chown=fastapi:fastapi docs/ /home/fastapi/docs/
RUN chown -R fastapi:fastapi /home/fastapi/docs/

COPY --chown=fastapi:fastapi entrypoint.sh .
RUN chmod +x /home/fastapi/entrypoint.sh

ENV PATH=/home/fastapi/.local/bin:$PATH

ENTRYPOINT ["/home/fastapi/entrypoint.sh"]

EXPOSE 8000
HEALTHCHECK CMD curl --fail http://localhost:8000/api/v1/health
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
