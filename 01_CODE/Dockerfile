ARG PYTHON_VERSION=python:3.12

# temp stage
FROM ${PYTHON_VERSION} AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update -y

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code
COPY src ./src
COPY setup.cfg setup.py pyproject.toml ./
COPY requirements.txt ./

RUN pip install -e .


FROM ${PYTHON_VERSION}-slim AS final

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --from=builder /app/src ./src
COPY main.py ./
COPY settings_main.cfg ./

# Create app user and directories with proper permissions
# Replace 996 with your actual docker group GID from the host
RUN groupadd -g 996 docker && \
    groupadd -r app && \
    useradd -r -g app -G docker -d /app app && \
    chown -R app:app /app

# RUN addgroup --gid 1001 --system app && \
#     adduser --no-create-home --shell /bin/false --disabled-password --uid 1001 --system --group app

USER app

# Set command
CMD ["/bin/bash", "-c", "whoami && ls -la && python main.py"]