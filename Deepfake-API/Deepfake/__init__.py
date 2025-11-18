"""
This is the root package for the Deepfake Detection API.
It ensures that the app directory is in the Python path.
"""
import os
import sys

# Add the app directory to the Python path
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, app_dir)

# Import the FastAPI app
from .app.main import app as web_app

__all__ = ['web_app']
