@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo 地震データ分析システム - スコア計算
echo モード: GNSSスコアあり
echo ========================================
echo.
echo 注意: GNSSデータは更新遅延があるため、
echo       最新データでは過小評価の可能性があります。
echo.

echo [1/2] スコアを計算中（GNSSあり）...
python ソース\calculate_scores.py
if errorlevel 1 (
    echo.
    echo エラー: スコア計算に失敗しました
    pause
    exit /b 1
)
echo.

echo [2/2] グラフを作成中...
python ソース\create_graphs.py
if errorlevel 1 (
    echo.
    echo エラー: グラフ作成に失敗しました
    pause
    exit /b 1
)
echo.

echo ========================================
echo スコア計算が完了しました（GNSSあり）
echo ========================================
echo.
echo スコア出力先:
echo   - データ分析結果出力\japan_scores.csv
echo   - データ分析結果出力\world_scores.csv
echo   - データ分析結果出力\depth_scores.csv
echo   - データ分析結果出力\gnss_scores.csv
echo   - データ分析結果出力\total_with_gnss.csv (国内+世界+深さ+GNSS)
echo.
echo グラフ出力先:
echo   - グラフ\total_score_timeseries_with_gnss.png
echo   - グラフ\total_score_histogram_with_gnss.png
echo   - グラフ\all_scores_comparison_with_gnss.png
echo.
pause
