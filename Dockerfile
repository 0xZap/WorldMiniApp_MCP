# Use a lightweight Python base image
FROM python:3.10-slim

# Stop Python from buffering output (useful for debugging/logging)
ENV PYTHONUNBUFFERED=1

# Create a directory in the container for our code
WORKDIR /app

# Copy in only requirements first (to leverage Docker build cache)
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the code
COPY . /app/

# Expose port 8080 (typical for GCP Cloud Run, though not strictly required)
EXPOSE 8080

# By default, Cloud Run sets PORT=8080, so our code uses os.environ.get("PORT", "8000").
# Launch the Python script
CMD ["python", "worldUI-MCP/worldUI_mcp.py"]