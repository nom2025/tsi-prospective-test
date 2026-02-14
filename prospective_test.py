"""
前向き検証（Prospective Test）評価スクリプト
============================================
凍結日: 2026-02-14
仕様書: TSI_FROZEN_SPEC.md

このスクリプトは凍結日以降のTSI値と地震データを用いて、
前向き検証の成績を評価する。

使い方:
    python prospective_test.py
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
from scipy import stats

BASE_DIR = Path(__file__).parent
FREEZE_DATE = pd.Timestamp("2026-02-14")
ALERT_THRESHOLD = 0.6138  # TSI_30d 95%ile（凍結）
M7_WINDOW = 60  # 日（凍結）


def load_data():
    """データ読み込み"""
    # TSI（凍結版計算結果）
    tsi_file = BASE_DIR / "データ分析結果出力" / "tsi_frozen.csv"
    if not tsi_file.exists():
        print("[エラー] tsi_frozen.csv が見つかりません。tsi_daily_calc.py を先に実行してください。")
        return None, None, None

    tsi = pd.read_csv(tsi_file, parse_dates=["date"], index_col="date")

    # 地震データ
    eq_jp = pd.read_csv(BASE_DIR / "生データ" / "2010-all.csv", parse_dates=["date"])
    eq_world = pd.read_csv(BASE_DIR / "生データ" / "world-all.csv", parse_dates=["date"])

    return tsi, eq_jp, eq_world


def evaluate_prospective(tsi, eq_jp, eq_world):
    """前向き検証の評価"""

    print("=" * 70)
    print("  前向き検証（Prospective Test）評価")
    print(f"  凍結日: {FREEZE_DATE.date()}")
    print(f"  評価日: {pd.Timestamp.now().date()}")
    print("=" * 70)

    # 凍結日以降のデータのみ
    tsi_prospective = tsi[tsi.index >= FREEZE_DATE]
    days_elapsed = (tsi_prospective.index.max() - FREEZE_DATE).days if len(tsi_prospective) > 0 else 0

    print(f"\n  検証期間: {FREEZE_DATE.date()} ～ {tsi_prospective.index.max().date() if len(tsi_prospective) > 0 else 'N/A'}")
    print(f"  経過日数: {days_elapsed}日")

    if days_elapsed < 30:
        print(f"\n  [注意] 検証期間が短すぎます（{days_elapsed}日）。最低30日必要です。")

    # --- 前向き期間の警報クラスタ ---
    alert_days = tsi_prospective[tsi_prospective["alert"] == True] if "alert" in tsi_prospective.columns else \
                 tsi_prospective[tsi_prospective["TSI_30d"] >= ALERT_THRESHOLD]

    if len(alert_days) == 0:
        print("\n  前向き期間に警報なし")
        clusters = []
    else:
        dates = sorted(alert_days.index.tolist())
        clusters = []
        cs = ce = dates[0]
        for d in dates[1:]:
            if (d - ce).days <= 5:
                ce = d
            else:
                clusters.append((cs, ce))
                cs = ce = d
        clusters.append((cs, ce))

    print(f"\n  前向き期間の警報クラスタ: {len(clusters)}")

    # --- 前向き期間のM7 ---
    m7_jp = eq_jp[(eq_jp["magnitude"] >= 7.0) & (eq_jp["date"] >= FREEZE_DATE)]
    m7_world = eq_world[(eq_world["magnitude_w"] >= 7.0) & (eq_world["date"] >= FREEZE_DATE)]

    print(f"  前向き期間のM7+ (日本): {len(m7_jp)}回")
    print(f"  前向き期間のM7+ (世界): {len(m7_world)}回")

    if len(m7_jp) > 0:
        print("\n  日本M7+一覧:")
        for _, r in m7_jp.iterrows():
            print(f"    {r['date'].date()} {r['place']} M{r['magnitude']}")

    if len(m7_world) > 0:
        print("\n  世界M7+一覧:")
        for _, r in m7_world.iterrows():
            print(f"    {r['date'].date()} {r['place_w']} M{r['magnitude_w']}")

    # --- 的中率評価（警報 → M7） ---
    print(f"\n{'─' * 70}")
    print("  ■ 的中率評価（警報 → M7）")
    print(f"{'─' * 70}")

    # M7を統合（日本+世界）
    m7_all_dates = set()
    if len(m7_jp) > 0:
        for d in m7_jp["date"].dt.date:
            m7_all_dates.add(d)
    if len(m7_world) > 0:
        for d in m7_world["date"].dt.date:
            m7_all_dates.add(d)

    hit_count = 0
    for i, (cs, ce) in enumerate(clusters):
        check_end = ce + timedelta(days=M7_WINDOW)
        peak_val = tsi_prospective.loc[cs:ce, "TSI_30d"].max() if "TSI_30d" in tsi_prospective.columns else np.nan
        peak_date = tsi_prospective.loc[cs:ce, "TSI_30d"].idxmax() if "TSI_30d" in tsi_prospective.columns else cs

        # M7チェック
        hit = False
        matched = []

        for _, r in m7_jp.iterrows():
            if cs <= r["date"] <= check_end:
                hit = True
                matched.append(f"{r['date'].date()} {r['place']} M{r['magnitude']}")

        for _, r in m7_world.iterrows():
            if cs <= r["date"] <= check_end:
                hit = True
                matched.append(f"{r['date'].date()} {r['place_w']} M{r['magnitude_w']}")

        if hit:
            hit_count += 1

        tag = "★的中" if hit else "空振り"
        duration = (ce - cs).days + 1
        print(f"\n  クラスタ{i+1}: {cs.date()} ～ {ce.date()} ({duration}日間) "
              f"TSI_30d max={peak_val:.4f} → {tag}")
        if matched:
            for m in matched:
                print(f"    → {m}")

    n_clusters = len(clusters)
    precision = hit_count / n_clusters if n_clusters > 0 else 0

    print(f"\n  的中率: {hit_count}/{n_clusters} = {precision:.4f} ({precision*100:.2f}%)")

    # --- 捕捉率評価（M7 → 警報） ---
    print(f"\n{'─' * 70}")
    print("  ■ 捕捉率評価（M7 → 警報）")
    print(f"{'─' * 70}")

    caught = 0
    total_m7 = 0

    for eq_data, eq_col, place_col in [(m7_jp, "magnitude", "place"), (m7_world, "magnitude_w", "place_w")]:
        for _, r in eq_data.iterrows():
            total_m7 += 1
            eq_date = r["date"]
            lookback = eq_date - timedelta(days=M7_WINDOW)
            prior_alerts = [c for c in clusters if c[0] >= lookback and c[0] <= eq_date]
            if prior_alerts:
                caught += 1
                print(f"  ✓ {eq_date.date()} {r[place_col]} M{r[eq_col]} ← 事前警報あり")
            else:
                print(f"  ✗ {eq_date.date()} {r[place_col]} M{r[eq_col]} ← 見逃し")

    recall = caught / total_m7 if total_m7 > 0 else 0
    print(f"\n  捕捉率: {caught}/{total_m7} = {recall:.4f} ({recall*100:.2f}%)")

    # --- ベースレート比較 ---
    print(f"\n{'─' * 70}")
    print("  ■ ベースレートとの比較")
    print(f"{'─' * 70}")

    # 過去のM7頻度からベースレート算出（凍結日以前のデータ）
    # 世界データのみ使用（日本データと重複するため）
    m7_world_past = eq_world[(eq_world["magnitude_w"] >= 7.0) & (eq_world["date"] < FREEZE_DATE)]
    past_days = (FREEZE_DATE - pd.Timestamp("2010-01-01")).days
    # ユニーク日数でカウント（同日の複数M7は1イベント）
    past_m7_dates = m7_world_past["date"].dt.date.nunique()
    p_daily = past_m7_dates / past_days if past_days > 0 else 0
    p_window = 1 - (1 - p_daily) ** M7_WINDOW

    print(f"  過去のM7頻度: {past_m7_dates}ユニーク日 / {past_days}日 = {p_daily:.6f}/日")
    print(f"  {M7_WINDOW}日ウィンドウのベースレート: {p_window:.4f} ({p_window*100:.2f}%)")
    print(f"  観測された的中率: {precision:.4f} ({precision*100:.2f}%)")

    if precision > 0 and p_window > 0:
        lift = precision / p_window
        print(f"  リフト: {lift:.2f}倍")
    else:
        print(f"  リフト: 計算不可")

    # --- 総合判定 ---
    print(f"\n{'=' * 70}")
    print("  ■ 総合判定")
    print(f"{'=' * 70}")

    if days_elapsed < 180:
        print(f"\n  状態: データ蓄積中（{days_elapsed}/{180}日）")
        print(f"  判定: 評価保留（最低6ヶ月必要）")
    else:
        if n_clusters == 0:
            print(f"\n  警報が出ていないため評価不可")
        elif precision > p_window * 2:
            print(f"\n  的中率がベースレートの2倍超 → 有望")
        elif precision > p_window:
            print(f"\n  的中率がベースレート超 → 要追加検証")
        else:
            print(f"\n  的中率がベースレート以下 → 現時点でシグナルなし")


def main():
    tsi, eq_jp, eq_world = load_data()
    if tsi is None:
        return
    evaluate_prospective(tsi, eq_jp, eq_world)


if __name__ == "__main__":
    main()
