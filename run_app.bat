@echo off
echo ============================================
echo    Photo Selection Web App
echo ============================================
echo.
echo Starting server...
echo Open http://localhost:5000 in your browser
echo.
echo Press Ctrl+C to stop the server
echo ============================================
echo.

cd /d "%~dp0"
python app.py

pause
