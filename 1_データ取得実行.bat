@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo 地震データ分析システム - データ取得
echo ========================================
echo.

cd データ自動取得

echo [1/4] 国内地震データを取得中...
python src\fetch_earthquake_data.py
if errorlevel 1 (
    echo.
    echo エラー: 国内地震データの取得に失敗しました
    pause
    exit /b 1
)
echo.

echo [2/4] 世界地震データを取得中...
python src\fetch_world_earthquake_data.py
if errorlevel 1 (
    echo.
    echo エラー: 世界地震データの取得に失敗しました
    pause
    exit /b 1
)
echo.

echo [3/4] 深さデータを取得中...
python src\fetch_depth_data.py
if errorlevel 1 (
    echo.
    echo エラー: 深さデータの取得に失敗しました
    pause
    exit /b 1
)
echo.

echo [4/4] GNSSデータを取得中...
python src\fetch_gnss_data.py
if errorlevel 1 (
    echo.
    echo エラー: GNSSデータの取得に失敗しました
    pause
    exit /b 1
)
echo.

cd ..

echo ========================================
echo データ取得が完了しました
echo ========================================
echo.
echo 出力先:
echo   - 生データ\2010-all.csv
echo   - 生データ\world-all.csv
echo   - 生データ\depth.csv
echo   - 生データ\gnss_data.csv
echo.
pause
