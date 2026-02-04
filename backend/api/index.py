import os
import sys
import logging
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import QueryRequest, QueryResponse, LoadDocumentsResponse, HealthResponse
from app.database import initialize_database, is_database_empty, get_document_count
from app.document_processor import scan_and_process_documents
from app.vector_store import store_embeddings, clear_all_chunks
from app.rag_chain import get_embedding, query_rag

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SOP RAG Agent API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and check if documents need to be loaded."""
    try:
        initialize_database()
        logger.info("Database initialized")
        
        # Check if database is empty and auto-load documents if needed
        # Note: For Vercel serverless, this runs on cold start
        # For Docker, this runs once when container starts
        if is_database_empty():
            logger.info("Database is empty, auto-loading documents...")
            data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            if os.path.exists(data_folder):
                # Run loading in background to avoid blocking startup
                asyncio.create_task(load_documents_async(data_folder))
            else:
                logger.warning(f"Data folder not found: {data_folder}")
        else:
            doc_count = get_document_count()
            logger.info(f"Database already contains {doc_count} document chunks")
    except Exception as e:
        logger.error(f"Error during startup: {e}")


async def load_documents_async(data_folder: str):
    """Async wrapper for loading documents."""
    try:
        load_documents_internal(data_folder)
        logger.info("Documents auto-loaded successfully")
    except Exception as e:
        logger.error(f"Error auto-loading documents: {e}")


def load_documents_internal(data_folder: str) -> tuple[int, int]:
    """Internal function to load documents. Returns (chunks_processed, files_processed)."""
    # Scan and process all documents
    all_chunks, files_processed = scan_and_process_documents(data_folder)
    
    if not all_chunks:
        return 0, files_processed
    
    # Get embeddings for all chunks
    logger.info(f"Getting embeddings for {len(all_chunks)} chunks...")
    embeddings = []
    for i, chunk in enumerate(all_chunks):
        if (i + 1) % 10 == 0:
            logger.info(f"Embedding progress: {i + 1}/{len(all_chunks)}")
        embedding = get_embedding(chunk['text'])
        embeddings.append(embedding)
    
    # Store in database
    logger.info("Storing embeddings in database...")
    store_embeddings(all_chunks, embeddings)
    
    return len(all_chunks), files_processed


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        from app.database import get_db_connection
        conn = get_db_connection()
        conn.close()
        database_connected = True
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        database_connected = False
    
    documents_loaded = not is_database_empty()
    
    return HealthResponse(
        status="healthy" if database_connected else "unhealthy",
        database_connected=database_connected,
        documents_loaded=documents_loaded
    )


@app.post("/api/load-documents", response_model=LoadDocumentsResponse)
async def load_documents():
    """Load documents from the data folder into the vector database."""
    try:
        data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        
        if not os.path.exists(data_folder):
            raise HTTPException(status_code=404, detail=f"Data folder not found: {data_folder}")
        
        # Clear existing chunks (optional - remove if you want to append)
        # clear_all_chunks()
        
        chunks_processed, files_processed = load_documents_internal(data_folder)
        
        return LoadDocumentsResponse(
            message="Documents loaded successfully",
            chunks_processed=chunks_processed,
            files_processed=files_processed
        )
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system."""
    try:
        result = query_rag(request.query, top_k=request.top_k)
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources']
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# For local development with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
