@echo off
setlocal
chcp 65001 >nul

set "ROOT=%~dp0"
set "BACKEND_PORT=8001"
set "FRONTEND_PORT=5174"
set "BACKEND_URL=http://127.0.0.1:%BACKEND_PORT%"
set "FRONTEND_URL=http://127.0.0.1:%FRONTEND_PORT%/"

echo BTCUSDT prediction dev launcher
echo Project: %ROOT%
echo Backend:  %BACKEND_URL%
echo Frontend: %FRONTEND_URL%
echo.

if not exist "%ROOT%.venv\Scripts\python.exe" (
  echo [ERROR] Missing Python virtualenv: %ROOT%.venv\Scripts\python.exe
  echo Create/install the virtualenv before running this launcher.
  pause
  exit /b 1
)

if not exist "%ROOT%frontend\package.json" (
  echo [ERROR] Missing frontend package.json: %ROOT%frontend\package.json
  pause
  exit /b 1
)

if not exist "%ROOT%frontend\node_modules" (
  echo [ERROR] Missing frontend dependencies: %ROOT%frontend\node_modules
  echo Run: cd frontend ^&^& npm install
  pause
  exit /b 1
)

if /i "%~1"=="--check" (
  echo [OK] Startup prerequisites found.
  exit /b 0
)

echo Starting backend window...
start "BTC Backend %BACKEND_PORT%" /D "%ROOT%" cmd /k "set PYTHONPATH=src&& .venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port %BACKEND_PORT%"

echo Starting frontend window...
start "BTC Frontend %FRONTEND_PORT%" /D "%ROOT%frontend" cmd /k "set VITE_API_BASE_URL=%BACKEND_URL%&& npm run dev -- --host 127.0.0.1 --port %FRONTEND_PORT%"

echo.
echo Waiting a few seconds before opening the browser...
timeout /t 5 /nobreak >nul
start "" "%FRONTEND_URL%"

echo.
echo Started. Close the backend/frontend command windows to stop the project.
exit /b 0
