@echo off
chcp 65001 > nul
echo パッケージをインストールしています...
.conda\python.exe -m pip install -r requirements.txt
echo.
echo インストールが完了しました。
pause

