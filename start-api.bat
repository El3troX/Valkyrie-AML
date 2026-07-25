@echo off
title Valkyrie-AML API Backend
echo Starting FastAPI Backend...
rem Set GROK_API_KEY here or set it in your environment
if "%GROK_API_KEY%"=="" set GROK_API_KEY=your_groq_api_key_here
set NROWS=200000
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
pause
