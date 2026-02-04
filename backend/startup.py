"""
Startup script that runs in Docker to automatically load documents if database is empty.
This script is called from the Dockerfile CMD.
"""
import os
import sys
import time
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Wait for the API to be ready
def wait_for_api(max_retries=30, delay=2):
    """Wait for the API server to be ready."""
    api_url = os.getenv("API_URL", "http://localhost:8000")
    health_url = f"{api_url}/api/health"
    
    for i in range(max_retries):
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                logger.info("API is ready")
                return True
        except Exception as e:
            logger.info(f"Waiting for API... (attempt {i+1}/{max_retries})")
            time.sleep(delay)
    
    logger.error("API did not become ready in time")
    return False


def load_documents():
    """Load documents via the API endpoint."""
    api_url = os.getenv("API_URL", "http://localhost:8000")
    load_url = f"{api_url}/api/load-documents"
    
    try:
        logger.info("Calling load-documents endpoint...")
        response = requests.post(load_url, timeout=3600)  # 1 hour timeout for large documents
        response.raise_for_status()
        result = response.json()
        logger.info(f"Documents loaded: {result.get('chunks_processed', 0)} chunks from {result.get('files_processed', 0)} files")
        return True
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return False


def check_and_load():
    """Check if database is empty and load documents if needed."""
    api_url = os.getenv("API_URL", "http://localhost:8000")
    health_url = f"{api_url}/api/health"
    
    try:
        response = requests.get(health_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if not data.get("documents_loaded", False):
                logger.info("Database is empty, loading documents...")
                return load_documents()
            else:
                logger.info("Documents already loaded, skipping...")
                return True
        else:
            logger.warning(f"Health check returned status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error checking database status: {e}")
        return False


if __name__ == "__main__":
    logger.info("Starting document loading check...")
    
    # Wait for API to be ready
    if not wait_for_api():
        logger.error("Failed to connect to API, exiting")
        sys.exit(1)
    
    # Check and load documents if needed
    if check_and_load():
        logger.info("Startup script completed successfully")
        sys.exit(0)
    else:
        logger.error("Startup script failed")
        sys.exit(1)
