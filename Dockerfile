FROM python:3.10-slim

# So Python logs aren't buffered
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies if needed (e.g. for scikit-learn)
RUN apt-get update && apt-get install -y curl build-essential

# Copy in pyproject.toml & poetry.lock first
COPY pyproject.toml poetry.lock* /app/

# Install Poetry
RUN pip install --upgrade pip
RUN pip install poetry

# Make Poetry install into global environment instead of .venv
RUN poetry config virtualenvs.create false

# Install Python dependencies
RUN poetry install --no-root --no-interaction --no-ansi

# Now copy the rest of your code
COPY . /app/

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Run the MCP builder server
CMD ["python", "worldBuilder/mcp_builder.py"]