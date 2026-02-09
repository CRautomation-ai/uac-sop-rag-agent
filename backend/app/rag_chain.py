import os
import logging
from typing import List, Dict, Optional
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


def rewrite_query_with_context(
    current_query: str,
    previous_messages: List[Dict[str, str]],
) -> str:
    """
    Use previous Q&A and current query to produce a standalone, context-rich question
    suitable for retrieval (embedding + search).
    """
    lines = []
    for pm in previous_messages:
        lines.append(f"Q: {pm.get('query', '')}\nA: {pm.get('answer', '')}")
    conv = "\n\n".join(lines)
    prompt = f"""Use the previous messages and the current user query to create a new, standalone user query that provides full context and is a complete question. The new query should be self-contained so that someone reading only it understands what is being asked. Output only the new query, nothing else.

Previous messages:
{conv}

Current user query: {current_query}

New standalone query:"""
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )
    rewritten = (response.choices[0].message.content or "").strip()
    return rewritten if rewritten else current_query


def query_rag(
    user_query: str,
    top_k: int = 5,
    previous_messages: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, any]:
    """
    Perform RAG query: embed query, search for similar chunks, generate answer.
    
    Args:
        user_query: User's question
        top_k: Number of similar chunks to retrieve
        previous_messages: Optional list of {"query": str, "answer": str} for conversation context
    
    Returns:
        Dictionary with answer and sources
    """
    try:
        # Step 0 (optional): Rewrite query with conversation context for better retrieval
        search_query = user_query
        if previous_messages:
            logger.info("Rewriting query with conversation context...")
            search_query = rewrite_query_with_context(user_query, previous_messages)
            logger.info(f"Rewritten query: {search_query[:80]}...")

        # Step 1: Embed the (rewritten) query
        logger.info(f"Embedding query: {search_query[:50]}...")
        query_embedding = get_embedding(search_query)

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

        # Step 4: Build prompt for OpenAI (use search_query so answer matches intent)
        system_prompt = """You are a helpful assistant named Bolt that answers questions based on the provided context from documents. 
Use only the information from the context to answer the question. If the context doesn't contain enough information to answer the question, say so."""

        user_prompt = f"""Context from documents:

{context}

Question: {search_query}
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
        
        answer = response.choices[0].message.content or ""

        return {
            'answer': answer,
            'sources': sources
        }
        
    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        raise
