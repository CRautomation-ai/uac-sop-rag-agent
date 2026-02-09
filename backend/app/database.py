import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging

logger = logging.getLogger(__name__)


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
    
    try:
        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info("pgvector extension enabled")
        
        # Create documents table with vector column
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                chunk_text TEXT NOT NULL,
                embedding vector(3072),
                source_file TEXT NOT NULL,
                folder_path TEXT,
                page_number INTEGER,
                chunk_index INTEGER,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create index for vector similarity search
        try:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
                ON document_chunks 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
        except Exception as e:
            # If ivfflat fails (e.g., no data yet), create a simple index
            logger.warning(f"Could not create ivfflat index: {e}. Will create after data is loaded.")
        
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
