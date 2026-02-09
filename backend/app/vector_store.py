import os
import logging
import json
from typing import List, Dict, Optional, Any
import psycopg2
from psycopg2.extras import execute_values
from app.database import get_db_connection

logger = logging.getLogger(__name__)


def store_embeddings(chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
    """
    Store document chunks and their embeddings in the database.
    
    Args:
        chunks: List of chunk dictionaries with metadata
        embeddings: List of embedding vectors (each is a list of floats)
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        values = []
        for chunk, embedding in zip(chunks, embeddings):
            # Convert embedding to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            # Remove null bytes from text (PostgreSQL doesn't allow them)
            chunk_text = chunk['text'].replace('\x00', '').strip()
            
            if not chunk_text:
                continue
            
            metadata_json = json.dumps(chunk.get('metadata', {}))
            
            values.append((
                chunk_text,
                embedding_str,
                chunk['source_file'],
                chunk.get('folder_path'),
                chunk.get('page_number'),
                chunk.get('chunk_index'),
                metadata_json
            ))
        
        # Batch insert
        insert_query = """
            INSERT INTO document_chunks 
            (chunk_text, embedding, source_file, folder_path, page_number, chunk_index, metadata)
            VALUES %s
        """
        
        execute_values(cursor, insert_query, values)
        conn.commit()
        
        logger.info(f"Stored {len(chunks)} chunks in database")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing embeddings: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def search_similar_chunks(
    query_embedding: List[float],
    top_k: int = 5,
    threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Search for similar chunks using cosine similarity.
    
    Args:
        query_embedding: Query embedding vector
        top_k: Number of results to return
        threshold: Minimum similarity threshold (0-1)
    
    Returns:
        List of similar chunks with metadata and similarity scores
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Convert embedding to PostgreSQL vector format
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Use cosine similarity (1 - cosine_distance)
        query = """
            SELECT 
                chunk_text,
                source_file,
                folder_path,
                page_number,
                chunk_index,
                metadata,
                1 - (embedding <=> %s::vector) as similarity
            FROM document_chunks
            WHERE 1 - (embedding <=> %s::vector) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        
        cursor.execute(query, (embedding_str, embedding_str, threshold, embedding_str, top_k))
        results = cursor.fetchall()
        
        chunks = []
        for row in results:
            metadata = row[5] if row[5] else {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            
            chunks.append({
                'text': row[0],
                'source_file': row[1],
                'folder_path': row[2],
                'page_number': row[3],
                'chunk_index': row[4],
                'metadata': metadata,
                'similarity': float(row[6])
            })
        
        logger.info(f"Found {len(chunks)} similar chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Error searching similar chunks: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
