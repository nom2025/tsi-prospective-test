"""
TSI ダッシュボード生成（GitHub Pages用）
========================================
prospective_log.csv と tsi_frozen.csv から
静的HTMLダッシュボードを生成する。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import base64
import io

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = PROJECT_ROOT / "_site"
FREEZE_DATE = "2026-02-14"
ALERT_THRESHOLD = 0.6138


def generate_tsi_chart_base64(tsi_df, log_df):
    """TSI時系列チャートをbase64 PNG文字列で返す"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

        # TSI_30d（全期間、薄い色）
        if "TSI_30d" in tsi_df.columns:
            recent = tsi_df[tsi_df.index >= "2023-01-01"]
            ax.plot(recent.index, recent["TSI_30d"], color="#cccccc", linewidth=0.8, label="TSI_30d (pre-freeze)")

        # 前向き検証期間（濃い色）
        if log_df is not None and len(log_df) > 0 and "TSI_30d" in log_df.columns:
            valid = log_df.dropna(subset=["TSI_30d"])
            if len(valid) > 0:
                ax.plot(valid.index, valid["TSI_30d"], color="#2196F3", linewidth=2, label="TSI_30d (prospective)")

        # 警報閾値
        ax.axhline(y=ALERT_THRESHOLD, color="#FF5722", linestyle="--", linewidth=1, alpha=0.7, label=f"Alert threshold ({ALERT_THRESHOLD})")

        # 凍結日
        ax.axvline(x=pd.Timestamp(FREEZE_DATE), color="#4CAF50", linestyle=":", linewidth=1, alpha=0.7, label=f"Freeze date ({FREEZE_DATE})")

        ax.set_ylabel("TSI_30d")
        ax.set_title("Tectonic Stress Index — Prospective Test")
        ax.legend(loc="upper left", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print(f"  チャート生成失敗: {e}")
        return None


def generate_html(tsi_df, log_df, eq_stats):
    """HTMLダッシュボードを生成"""

    now = datetime.now().strftime("%Y-%m-%d %H:%M JST")
    freeze_date = FREEZE_DATE
    days_elapsed = (datetime.now() - datetime(2026, 2, 14)).days

    # 最新TSI値
    latest_tsi = "N/A"
    latest_tsi_30d = "N/A"
    latest_date = "N/A"
    alert_status = "データなし"
    alert_class = "status-unknown"

    if log_df is not None and len(log_df) > 0:
        latest = log_df.iloc[-1]
        latest_date = str(log_df.index[-1].date()) if hasattr(log_df.index[-1], 'date') else str(log_df.index[-1])
        latest_tsi = f"{latest.get('TSI', 'N/A'):.4f}" if pd.notna(latest.get('TSI')) else "N/A"
        latest_tsi_30d = f"{latest.get('TSI_30d', 'N/A'):.4f}" if pd.notna(latest.get('TSI_30d')) else "N/A"

        if pd.notna(latest.get('TSI_30d')):
            if latest['TSI_30d'] >= ALERT_THRESHOLD:
                alert_status = "⚠ 警報中"
                alert_class = "status-alert"
            else:
                alert_status = "通常"
                alert_class = "status-normal"

    # チャート
    chart_b64 = generate_tsi_chart_base64(tsi_df, log_df)
    chart_html = f'<img src="data:image/png;base64,{chart_b64}" style="width:100%;max-width:900px;">' if chart_b64 else "<p>チャート生成不可</p>"

    # ログテーブル
    log_table_rows = ""
    if log_df is not None and len(log_df) > 0:
        recent = log_df.tail(30).iloc[::-1]  # 最新30日、降順
        for idx, row in recent.iterrows():
            date_str = str(idx.date()) if hasattr(idx, 'date') else str(idx)
            tsi_val = f"{row['TSI']:.4f}" if pd.notna(row.get('TSI')) else "-"
            tsi_30d_val = f"{row['TSI_30d']:.4f}" if pd.notna(row.get('TSI_30d')) else "-"
            is_alert = pd.notna(row.get('TSI_30d')) and row['TSI_30d'] >= ALERT_THRESHOLD
            row_class = 'class="alert-row"' if is_alert else ""
            alert_mark = "⚠" if is_alert else ""
            log_table_rows += f"<tr {row_class}><td>{date_str}</td><td>{tsi_val}</td><td>{tsi_30d_val}</td><td>{alert_mark}</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TSI Prospective Test Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 1.5em; border-bottom: 2px solid #333; padding-bottom: 8px; }}
  .card {{ background: white; border-radius: 8px; padding: 20px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
  .metric {{ text-align: center; }}
  .metric .value {{ font-size: 2em; font-weight: bold; }}
  .metric .label {{ font-size: 0.85em; color: #666; }}
  .status-normal .value {{ color: #4CAF50; }}
  .status-alert .value {{ color: #FF5722; }}
  .status-unknown .value {{ color: #999; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  th, td {{ padding: 6px 12px; text-align: right; border-bottom: 1px solid #eee; }}
  th {{ background: #f9f9f9; font-weight: 600; text-align: right; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .alert-row {{ background: #FFF3E0; }}
  .footer {{ text-align: center; font-size: 0.8em; color: #999; margin-top: 32px; }}
  .frozen-badge {{ display: inline-block; background: #E8F5E9; color: #2E7D32; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: bold; }}
</style>
</head>
<body>
<div class="container">

<h1>Tectonic Stress Index (TSI) <span class="frozen-badge">FROZEN {freeze_date}</span></h1>

<div class="card">
  <div class="metrics">
    <div class="metric {alert_class}">
      <div class="value">{alert_status}</div>
      <div class="label">現在の状態</div>
    </div>
    <div class="metric">
      <div class="value">{latest_tsi_30d}</div>
      <div class="label">TSI_30d（最新）</div>
    </div>
    <div class="metric">
      <div class="value">{latest_tsi}</div>
      <div class="label">TSI（日次）</div>
    </div>
    <div class="metric">
      <div class="value">{days_elapsed}</div>
      <div class="label">検証経過日数</div>
    </div>
  </div>
  <p style="font-size:0.85em;color:#666;margin-top:12px;">最終更新: {latest_date} | 警報閾値: TSI_30d ≥ {ALERT_THRESHOLD}</p>
</div>

<div class="card">
  <h2 style="margin-top:0;">TSI 時系列</h2>
  {chart_html}
</div>

<div class="card">
  <h2 style="margin-top:0;">前向き検証ログ（直近30日）</h2>
  <table>
    <thead><tr><th>日付</th><th>TSI</th><th>TSI_30d</th><th>警報</th></tr></thead>
    <tbody>
      {log_table_rows if log_table_rows else '<tr><td colspan="4" style="text-align:center;color:#999;">データなし</td></tr>'}
    </tbody>
  </table>
</div>

<div class="footer">
  <p>TSI Prospective Test — 凍結日: {freeze_date} — 自動更新: GitHub Actions</p>
  <p>このダッシュボードは前向き検証の透明性を確保するために自動生成されています。</p>
  <p>最終生成: {now}</p>
</div>

</div>
</body>
</html>"""

    return html


def main():
    print("=" * 50)
    print("  TSI Dashboard 生成")
    print("=" * 50)

    SITE_DIR.mkdir(exist_ok=True)

    # TSI読み込み
    tsi_file = PROJECT_ROOT / "データ分析結果出力" / "tsi_frozen.csv"
    tsi_df = None
    if tsi_file.exists():
        tsi_df = pd.read_csv(tsi_file, parse_dates=["date"], index_col="date")
        print(f"  tsi_frozen.csv: {len(tsi_df)}行")

    # 前向き検証ログ
    log_file = PROJECT_ROOT / "prospective_log.csv"
    log_df = None
    if log_file.exists():
        log_df = pd.read_csv(log_file, parse_dates=["date"], index_col="date")
        print(f"  prospective_log.csv: {len(log_df)}行")

    # HTML生成
    html = generate_html(tsi_df, log_df, {})

    output_file = SITE_DIR / "index.html"
    output_file.write_text(html, encoding="utf-8")
    print(f"  出力: {output_file}")
    print("  完了")


if __name__ == "__main__":
    main()
