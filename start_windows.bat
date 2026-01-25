@echo off
setlocal
cd /d %~dp0

REM 1) Install dependencies (first run only)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM 2) Open browser
start http://127.0.0.1:5000

REM 3) Run server
python app.py
