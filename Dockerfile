# Builder Stage: Install all dependencies
FROM docker-prod.artifactory.tmna-devops.com/python:3.12-slim AS builder

WORKDIR /app

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY dist/pip-packages /tmp/pip-packages


# Install Python dependencies and copy required libraries securely
RUN python -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --no-index --no-build-isolation --find-links=/tmp/pip-packages/ /tmp/pip-packages/* && \
    mkdir -p /venv/lib && \
    find /venv/bin -type f -executable -exec file {} \; | awk -F: '/ELF/ {print $1}' | \
    while IFS= read -r binary; do \
        ldd "$binary" | awk '/=>/ {print $3}' | grep -v "^(" | xargs -r -I{} cp --parents "{}" /venv/lib; \
    done

FROM docker-prod.artifactory.tmna-devops.com/tdsp/tdsp-mlops-wandb:1.0.4 AS wandb
WORKDIR /opt/ml/code
COPY setup.py .

FROM docker-prod.artifactory.tmna-devops.com/coc/chofer-golden-containers/language-os/amazonlinux2023-python3.12:1.3.0

# Set working directory
WORKDIR /opt/ml/code

# Copy the application code
COPY . .

# Copy the virtual environment from the builder stage
COPY --from=builder /venv /venv
COPY --from=wandb /opt/ml/code/src/initialize_tdspds.sh src/initialize_tdspds.sh
COPY --from=wandb /opt/ml/code/src/initiate_wandb.py src/initiate_wandb.py
COPY --from=wandb /opt/ml/code/src/utils src/utils

# Set environment variables for Python and shared libraries
ENV PATH="/venv/bin:$PATH"
ENV PYTHONPATH="/opt/ml/code/src:/venv/lib/python3.12/site-packages:$PYTHONPATH"
ENV LD_LIBRARY_PATH="/venv/lib:$LD_LIBRARY_PATH"

USER root

# Ensure 'python' and 'python3' are set to Python 3.12
RUN ln -sf "$(command -v python3.12)" /usr/bin/python && \
    ln -sf "$(command -v python3.12)" /usr/bin/python3

# Ensure entrypoint script is executable
RUN chmod +x src/main_script.sh && \
    chmod +x src/initialize_tdspds.sh

RUN id -u nonroot >/dev/null 2>&1 || \
    adduser -D nonroot && \
    echo 'nonroot ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Change ownership
RUN chown -R nonroot:nonroot /opt/ml/code /venv/lib/python3.12/site-packages

# # Switch to non-root user
USER nonroot

# Set the entrypoint
ENTRYPOINT ["src/main_script.sh"]