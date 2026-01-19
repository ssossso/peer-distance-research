#!/bin/bash
cd "$(dirname "$0")"
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
open http://127.0.0.1:5000
python3 app.py
