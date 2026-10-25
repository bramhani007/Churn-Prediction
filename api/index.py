import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

# Vercel serverless function entrypoint
def handler(request, response):
      return app(request, response)

# For direct execution/WSGI compatibility
app_handler = app
