import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging

logger = logging.getLogger(__name__)
IVFFLAT_MAX_DIMENSIONS = 2000


def get_embedding_dimensions() -> int:
    """Get embedding dimension from env, defaulting to text-embedding-3-large size."""
    raw = os.getenv("EMBEDDING_DIM", "3072")
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid EMBEDDING_DIM '%s'; defaulting to 3072", raw)
        value = 3072
    return value


def get_db_connection():
    """Get database connection from environment variable."""
    postgres_url = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")
    if not postgres_url:
        raise ValueError("POSTGRES_URL or DATABASE_URL environment variable is required")
    return psycopg2.connect(postgres_url)


def initialize_database():
    """Initialize database: create pgvector extension and tables if they don't exist."""
    conn = get_db_connection()
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    embedding_dimensions = get_embedding_dimensions()
    
    try:
        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info("pgvector extension enabled")
        
        # Create documents table with vector column
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                chunk_text TEXT NOT NULL,
                embedding vector(%s),
                source_file TEXT NOT NULL,
                folder_path TEXT,
                page_number INTEGER,
                chunk_index INTEGER,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """, (embedding_dimensions,))
        
        # Create index for vector similarity search
        if embedding_dimensions <= IVFFLAT_MAX_DIMENSIONS:
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
                    ON document_chunks
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
            except Exception as e:
                logger.warning(f"Could not create ivfflat index: {e}")
        else:
            logger.info(
                "Skipping ivfflat index: embedding dim %s exceeds supported max %s",
                embedding_dimensions,
                IVFFLAT_MAX_DIMENSIONS
            )
        
        # Create index for source file lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_source_file_idx 
            ON document_chunks (source_file);
        """)
        
        conn.commit()
        logger.info("Database tables initialized")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def is_database_empty():
    """Check if the database has any document chunks."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT COUNT(*) FROM document_chunks;")
        count = cursor.fetchone()[0]
        return count == 0
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        return True
    finally:
        cursor.close()
        conn.close()


def get_document_count():
    """Get the number of document chunks in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT COUNT(*) FROM document_chunks;")
        count = cursor.fetchone()[0]
        return count
    except Exception as e:
        logger.error(f"Error getting document count: {e}")
        return 0
    finally:
        cursor.close()
        conn.close()
