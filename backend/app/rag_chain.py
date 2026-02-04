import os
import logging
import re
from typing import List, Dict
from openai import OpenAI
from app.vector_store import search_similar_chunks

logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")


def get_embedding(text: str) -> List[float]:
    try:
        response = openai_client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        raise


def format_source_citation(chunk: Dict[str, any]) -> str:
    """Format a chunk into a source citation string."""
    parts = []
    
    if chunk.get('folder_path'):
        parts.append(chunk['folder_path'])
    
    parts.append(chunk['source_file'])
    
    if chunk.get('page_number'):
        parts.append(f"Page {chunk['page_number']}")
    
    return " > ".join(parts)


def query_rag(user_query: str, top_k: int = 5) -> Dict[str, any]:
    """
    Perform RAG query: embed query, search for similar chunks, generate answer.
    
    Args:
        user_query: User's question
        top_k: Number of similar chunks to retrieve
    
    Returns:
        Dictionary with answer and sources
    """
    try:
        # Step 1: Embed the query
        logger.info(f"Embedding query: {user_query[:50]}...")
        query_embedding = get_embedding(user_query)
        
        # Step 2: Search for similar chunks
        logger.info(f"Searching for similar chunks (top_k={top_k})...")
        similar_chunks = search_similar_chunks(query_embedding, top_k=top_k)
        
        if not similar_chunks:
            return {
                'answer': "I couldn't find any relevant information in the documents to answer your question.",
                'sources': []
            }
        
        # Step 3: Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for chunk in similar_chunks:
            context_parts.append(f"[Source: {format_source_citation(chunk)}]\n{chunk['text']}")
            source_citation = format_source_citation(chunk)
            if source_citation not in sources:
                sources.append(source_citation)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Step 4: Build prompt for OpenAI
        system_prompt = """You are a helpful assistant that answers questions based on the provided context from documents. 
Use only the information from the context to answer the question. If the context doesn't contain enough information to answer the question, say so."""
        
        user_prompt = f"""Context from documents: {context}
        Question: {user_query}
"""
        
        # Step 5: Call OpenAI to generate answer
        logger.info("Generating answer with OpenAI...")
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        # Remove any markdown formatting that might still be present
        # Remove markdown code blocks
        answer = re.sub(r'```[\s\S]*?```', '', answer)
        # Remove markdown bold/italic
        answer = re.sub(r'\*\*([^\*]+)\*\*', r'\1', answer)
        answer = re.sub(r'\*([^\*]+)\*', r'\1', answer)
        answer = re.sub(r'__([^_]+)__', r'\1', answer)
        answer = re.sub(r'_([^_]+)_', r'\1', answer)
        # Remove markdown headers
        answer = re.sub(r'^#+\s+', '', answer, flags=re.MULTILINE)
        # Remove markdown links but keep text
        answer = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', answer)
        # Clean up extra whitespace
        answer = answer.strip()
        
        return {
            'answer': answer,
            'sources': sources
        }
        
    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        raise
