import os
import sys
import logging
import asyncio
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add backend directory to path so "from app.*" resolves to backend/app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models import AuthRequest, AuthResponse, QueryRequest, QueryResponse, LoadDocumentsResponse, HealthResponse
from app.database import initialize_database, is_database_empty, get_document_count
from app.rag_chain import query_rag
from app.auth import verify_password, create_token, get_current_token

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SOP RAG Agent API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize database and check if documents need to be loaded."""
    try:
        initialize_database()
        logger.info("Database initialized")

        # Check if database is empty and auto-load documents if needed
        if is_database_empty():
            logger.info("Database is empty, auto-loading documents...")
            data_folder = os.path.join(os.path.dirname(__file__), "data")
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
    from app.document_processor import scan_and_process_documents
    from app.rag_chain import get_embedding
    from app.vector_store import store_embeddings

    all_chunks, files_processed = scan_and_process_documents(data_folder)

    if not all_chunks:
        return 0, files_processed

    logger.info(f"Getting embeddings for {len(all_chunks)} chunks...")
    embeddings = []
    for i, chunk in enumerate(all_chunks):
        if (i + 1) % 10 == 0:
            logger.info(f"Embedding progress: {i + 1}/{len(all_chunks)}")
        embedding = get_embedding(chunk['text'])
        embeddings.append(embedding)

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


@app.post("/api/auth", response_model=AuthResponse)
@app.post("/auth", response_model=AuthResponse)
async def auth(request: AuthRequest):
    """Verify password and return JWT. Used by both /api/auth and /auth for local proxy."""
    print('hi')
    if not verify_password(request.password):
        raise HTTPException(status_code=401, detail="Invalid password")
    return AuthResponse(token=create_token())


@app.post("/api/load-documents", response_model=LoadDocumentsResponse)
async def load_documents():
    """Load documents from the data folder into the vector database."""
    try:
        data_folder = os.path.join(os.path.dirname(__file__), "data")

        if not os.path.exists(data_folder):
            raise HTTPException(status_code=404, detail=f"Data folder not found: {data_folder}")

        chunks_processed, files_processed = load_documents_internal(data_folder)

        return LoadDocumentsResponse(
            message="Documents loaded successfully",
            chunks_processed=chunks_processed,
            files_processed=files_processed
        )
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _handle_query(request: QueryRequest):
    """Shared query logic for /api/query and /query."""
    prev = None
    if request.previous_messages:
        prev = [{"query": m.query, "answer": m.answer} for m in request.previous_messages]
    result = query_rag(request.query, top_k=request.top_k, previous_messages=prev)
    return QueryResponse(
        answer=result['answer'],
        sources=result['sources']
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest, _: str = Depends(get_current_token)):
    """Query the RAG system (protected)."""
    try:
        return _handle_query(request)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_local(request: QueryRequest, _: str = Depends(get_current_token)):
    """Query endpoint for local dev when proxy strips /api prefix."""
    try:
        return _handle_query(request)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# For local development with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
