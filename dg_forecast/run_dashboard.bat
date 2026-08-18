@echo off
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
chcp 65001 > nul
title Hassi R'Mel SCADA Gas Flow Forecasting Control Room
echo ====================================================================
echo Starting Hassi R'Mel SCADA Industrial Gas Flow Forecasting Dashboard
echo Environment: Conda well_predict
echo ====================================================================
"C:\Users\BADRO INFO\.conda\envs\well_predict\python.exe" -m streamlit run app.py
pause
