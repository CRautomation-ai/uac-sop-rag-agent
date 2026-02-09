import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
from docx import Document
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Initialize tokenizer for token counting
encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(encoding.encode(text))


def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF file, returning pages with text and page numbers."""
    pages = []
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                if text.strip():
                    pages.append({
                        'text': text,
                        'page_number': page_num
                    })
        logger.info(f"Extracted {len(pages)} pages from PDF: {file_path}")
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {e}")
        raise
    return pages


def extract_text_from_docx(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from Word document."""
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        
        text = '\n\n'.join(full_text)
        if text.strip():
            return [{
                'text': text,
                'page_number': None  # Word docs don't have clear page numbers
            }]
        logger.info(f"Extracted text from DOCX: {file_path}")
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path}: {e}")
        raise
    return []


def chunk_text_by_tokens(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    respect_sentences: bool = True
) -> List[str]:
    """
    Chunk text by tokens, respecting paragraph and sentence boundaries.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        respect_sentences: Whether to respect sentence boundaries
    
    Returns:
        List of text chunks
    """
    # Use RecursiveCharacterTextSplitter which respects paragraph/sentence boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""]  # Try to split at paragraphs, then sentences
    )
    
    chunks = text_splitter.split_text(text)
    return chunks


def process_document(file_path: str, base_path: str) -> List[Dict[str, Any]]:
    """
    Process a single document (PDF or Word) and return chunks with metadata.
    
    Args:
        file_path: Full path to the document
        base_path: Base path to calculate relative folder path
    
    Returns:
        List of chunks with metadata
    """
    file_ext = Path(file_path).suffix.lower()
    relative_path = os.path.relpath(file_path, base_path)
    folder_path = str(Path(relative_path).parent) if Path(relative_path).parent != Path('.') else None
    
    chunks = []
    
    try:
        if file_ext == '.pdf':
            pages = extract_text_from_pdf(file_path)
            for page_data in pages:
                page_chunks = chunk_text_by_tokens(page_data['text'])
                for idx, chunk_text in enumerate(page_chunks):
                    chunks.append({
                        'text': chunk_text,
                        'source_file': os.path.basename(file_path),
                        'folder_path': folder_path,
                        'page_number': page_data['page_number'],
                        'chunk_index': idx,
                        'metadata': {
                            'file_path': relative_path,
                            'file_type': 'pdf'
                        }
                    })
        
        elif file_ext in ['.docx', '.doc']:
            pages = extract_text_from_docx(file_path)
            for page_data in pages:
                page_chunks = chunk_text_by_tokens(page_data['text'])
                for idx, chunk_text in enumerate(page_chunks):
                    chunks.append({
                        'text': chunk_text,
                        'source_file': os.path.basename(file_path),
                        'folder_path': folder_path,
                        'page_number': page_data['page_number'],
                        'chunk_index': idx,
                        'metadata': {
                            'file_path': relative_path,
                            'file_type': 'docx'
                        }
                    })
        
        logger.info(f"Processed {file_path}: {len(chunks)} chunks created")
        
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {e}")
        # Continue processing other documents even if one fails
    
    return chunks


def scan_and_process_documents(data_folder: str) -> tuple[List[Dict[str, Any]], int]:
    """
    Recursively scan data folder for PDFs and Word docs, process them, and return all chunks.
    
    Args:
        data_folder: Path to the data folder
    
    Returns:
        Tuple of (list of all document chunks with metadata, number of files processed)
    """
    all_chunks = []
    data_path = Path(data_folder)
    
    if not data_path.exists():
        logger.error(f"Data folder does not exist: {data_folder}")
        return (all_chunks, 0)
    
    supported_extensions = {'.pdf', '.docx', '.doc'}
    
    # Recursively find all PDF and Word documents
    files_processed = 0
    for file_path in data_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            logger.info(f"Processing: {file_path}")
            chunks = process_document(str(file_path), str(data_path))
            all_chunks.extend(chunks)
            files_processed += 1
    
    logger.info(f"Total files processed: {files_processed}, Total chunks created: {len(all_chunks)}")
    return all_chunks, files_processed
