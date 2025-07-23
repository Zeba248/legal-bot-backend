#!/usr/bin/env python3
"""
Legal AI Chatbot - Deployment Script
This script handles the complete deployment and initialization of the legal chatbot system.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description=""):
    """
    Run a shell command and handle errors
    """
    logger.info(f"Running: {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        return False

def check_python_version():
    """
    Check if Python version is compatible
    """
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def setup_virtual_environment():
    """
    Create and activate virtual environment
    """
    if not os.path.exists("venv"):
        logger.info("Creating virtual environment...")
        if not run_command("python3 -m venv venv", "Creating virtual environment"):
            return False
    
    logger.info("Virtual environment ready")
    return True

def install_dependencies():
    """
    Install required Python packages
    """
    logger.info("Installing dependencies...")
    activate_cmd = "source venv/bin/activate" if os.name != 'nt' else "venv\\Scripts\\activate"
    install_cmd = f"{activate_cmd} && pip install --upgrade pip && pip install -r requirements.txt"
    
    return run_command(install_cmd, "Installing Python dependencies")

def check_data_files():
    """
    Check if required PDF files are present
    """
    data_dir = Path("Data")
    if not data_dir.exists():
        logger.error("Data directory not found")
        return False
    
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in Data directory")
        return False
    
    logger.info(f"Found {len(pdf_files)} PDF files: {[f.name for f in pdf_files]}")
    return True

def initialize_vector_database():
    """
    Initialize the vector database with legal documents
    """
    logger.info("Initializing vector database...")
    activate_cmd = "source venv/bin/activate" if os.name != 'nt' else "venv\\Scripts\\activate"
    init_cmd = f"{activate_cmd} && python3 initialize_data.py"
    
    return run_command(init_cmd, "Initializing vector database")

def test_system():
    """
    Run basic system tests
    """
    logger.info("Testing system components...")
    activate_cmd = "source venv/bin/activate" if os.name != 'nt' else "venv\\Scripts\\activate"
    
    # Test imports
    test_cmd = f"{activate_cmd} && python3 -c 'from main import app; print(\"✅ Main application imports successful\")'"
    if not run_command(test_cmd, "Testing imports"):
        return False
    
    logger.info("✅ System tests passed")
    return True

def create_env_file():
    """
    Create .env file from example if it doesn't exist
    """
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        logger.info("Creating .env file from example...")
        with open(".env.example", "r") as src, open(".env", "w") as dst:
            dst.write(src.read())
        logger.info("✅ Created .env file. Please update it with your API keys.")
        return True
    elif os.path.exists(".env"):
        logger.info("✅ .env file already exists")
        return True
    else:
        logger.warning("⚠️  No .env.example file found")
        return True

def main():
    """
    Main deployment function
    """
    logger.info("🚀 Starting Legal AI Chatbot Deployment")
    logger.info("=" * 50)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Setting up virtual environment", setup_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Creating environment file", create_env_file),
        ("Checking data files", check_data_files),
        ("Initializing vector database", initialize_vector_database),
        ("Testing system", test_system),
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\n📋 Step: {step_name}")
        if not step_func():
            logger.error(f"❌ Failed at step: {step_name}")
            sys.exit(1)
        logger.info(f"✅ Completed: {step_name}")
    
    logger.info("\n" + "=" * 50)
    logger.info("🎉 Deployment completed successfully!")
    logger.info("\n📚 Next Steps:")
    logger.info("1. Update .env file with your API keys (optional for basic functionality)")
    logger.info("2. Start the server: python3 main.py")
    logger.info("3. Test the API:")
    logger.info("   - System info: curl http://localhost:8000/system-info")
    logger.info("   - Search docs: curl -X POST http://localhost:8000/search -H 'Content-Type: application/json' -d '{\"question\": \"What is theft?\"}'")
    logger.info("   - Full RAG (needs API keys): curl -X POST http://localhost:8000/ask -H 'Content-Type: application/json' -d '{\"question\": \"What is theft?\"}'")
    logger.info("\n🌐 The system is ready for production deployment!")

if __name__ == "__main__":
    main()