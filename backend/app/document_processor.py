import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
from docx import Document
import tiktoken

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc'}

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
    chunk_overlap: int = 200
) -> List[str]:
    """
    Chunk text by tokens, respecting paragraph and sentence boundaries.
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
    Returns:
        List of text chunks
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    tokens = encoding.encode(clean_text)
    if not tokens:
        return []

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)

    while start < len(tokens):
        token_slice = tokens[start:start + chunk_size]
        chunk = encoding.decode(token_slice).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(tokens):
            break
        start += step

    return chunks


def process_document(
    file_path: str,
    base_path: str,
    relative_path_override: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Process a single document (PDF or Word) and return chunks with metadata.
    
    Args:
        file_path: Full path to the document
        base_path: Base path to calculate relative folder path
    
    Returns:
        List of chunks with metadata
    """
    file_ext = Path(file_path).suffix.lower()
    relative_path = relative_path_override or os.path.relpath(file_path, base_path)
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
    
    # Recursively find all PDF and Word documents
    files_processed = 0
    for file_path in data_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            logger.info(f"Processing: {file_path}")
            chunks = process_document(str(file_path), str(data_path))
            all_chunks.extend(chunks)
            files_processed += 1
    
    logger.info(f"Total files processed: {files_processed}, Total chunks created: {len(all_chunks)}")
    return all_chunks, files_processed


def scan_and_process_file_paths(
    file_entries: List[Dict[str, str]],
    base_path: str
) -> tuple[List[Dict[str, Any]], int]:
    """
    Process uploaded files from explicit file paths.

    Args:
        file_entries: List of {"path": absolute_path, "relative_path": original_upload_path}
        base_path: Fallback base path for relative path calculation

    Returns:
        Tuple of (all chunks, files processed)
    """
    all_chunks = []
    files_processed = 0

    for entry in file_entries:
        file_path = entry["path"]
        relative_path = entry.get("relative_path")
        ext = Path(file_path).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        logger.info(f"Processing uploaded file: {file_path}")
        chunks = process_document(
            file_path=file_path,
            base_path=base_path,
            relative_path_override=relative_path
        )
        all_chunks.extend(chunks)
        files_processed += 1

    logger.info(
        "Uploaded files processed: %s, Total chunks created: %s",
        files_processed,
        len(all_chunks)
    )
    return all_chunks, files_processed
