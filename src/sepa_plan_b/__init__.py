"""SEPA Plan B. No dependencies on legacy research packages."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
__version__ = "0.1.0"
