@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo 地震データ分析システム - スコア計算
echo モード: GNSSスコアなし
echo ========================================
echo.
echo 注意: GNSSを除外したスコアです。
echo       地震データのみで評価します。
echo.

echo [1/2] スコアを計算中（GNSSなし）...
python ソース\calculate_scores.py --no-gnss
if errorlevel 1 (
    echo.
    echo エラー: スコア計算に失敗しました
    pause
    exit /b 1
)
echo.

echo [2/2] グラフを作成中...
python ソース\create_graphs.py --no-gnss
if errorlevel 1 (
    echo.
    echo エラー: グラフ作成に失敗しました
    pause
    exit /b 1
)
echo.

echo ========================================
echo スコア計算が完了しました（GNSSなし）
echo ========================================
echo.
echo スコア出力先:
echo   - データ分析結果出力\japan_scores.csv
echo   - データ分析結果出力\world_scores.csv
echo   - データ分析結果出力\depth_scores.csv
echo   - データ分析結果出力\total_no_gnss.csv (国内+世界+深さのみ)
echo.
echo グラフ出力先:
echo   - グラフ\total_score_timeseries_no_gnss.png
echo   - グラフ\total_score_histogram_no_gnss.png
echo   - グラフ\all_scores_comparison_no_gnss.png
echo.
pause
