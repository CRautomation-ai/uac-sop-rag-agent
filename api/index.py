"""
Vercel serverless entry: delegates to the FastAPI app in backend.
Ensures repo root is on path so "from backend.api.index import app" works.
"""
import os
import sys

_repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _repo_root)

from backend.api.index import app
