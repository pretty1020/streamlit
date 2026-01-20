"""
Forecast_ML.py - Streamlit Application Entry Point

IMPORTANT: This is a STREAMLIT application, NOT a Shiny application.
The application uses streamlit library, NOT shiny.
All dependencies are listed in requirements.txt - shiny is NOT required.
"""

# CRITICAL: Import streamlit FIRST to explicitly identify this as a Streamlit app
# This helps deployment systems correctly identify the application type
import streamlit as st
import sys
import os

# Add current directory to Python path to ensure models can be found
# This is critical for deployment environments where the working directory may differ
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import and execute main.py
# This ensures all imports work correctly in deployment
# NOTE: This is a Streamlit app - no shiny dependencies needed
exec(open(os.path.join(current_dir, 'main.py'), encoding='utf-8').read())
