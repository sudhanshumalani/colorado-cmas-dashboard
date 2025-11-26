@echo off
echo ========================================
echo Installing RAG System Dependencies
echo ========================================
echo.

echo Installing core dependencies first...
pip install streamlit anthropic pandas openpyxl python-dotenv plotly numpy scipy

echo.
echo Installing database...
pip install duckdb sqlalchemy

echo.
echo Installing vector store (this may take a few minutes)...
pip install chromadb sentence-transformers

echo.
echo Installing optional dependencies...
pip install streamlit-chat redis

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Next steps:
echo 1. Create .env file with your ANTHROPIC_API_KEY
echo 2. Run: python setup_rag.py
echo 3. Run: streamlit run app_rag.py
echo.
pause
