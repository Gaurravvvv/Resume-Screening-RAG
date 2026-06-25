@echo off
echo Checking dependencies...
python -c "import streamlit, openai, langchain, pypdf, streamlit_modal" >nul 2>&1
if %errorlevel% neq 0 (
    echo Some dependencies are missing. Installing required packages...
    python -m pip install langchain-community langchain-openai streamlit-modal sentence-transformers faiss-cpu pypdf
) else (
    echo All dependencies are satisfied!
)

echo.
echo Starting Resume Screening Chatbot...
echo The web application will automatically open in your default browser shortly.
python -m streamlit run demo/interface.py --server.headless false
pause
