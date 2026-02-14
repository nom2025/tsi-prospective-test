"""
循環論法テスト & 成分分離分析
==============================
問題: total_score の構成要素に地震データそのものが含まれている

total_score = japan_score + world_score + depth_score + gnss_score + b_value_score

地震データに依存するもの（循環）:
  - japan_score: 国内地震の回数・規模・場所を直接使用
  - world_score: 世界の地震M6+の回数・規模を直接使用
  - depth_score: 深発地震データを使用
  - b_value_score: 国内地震のb値（マグニチュード分布）を使用

地震データに依存しないもの（独立）:
  - gnss_score: GNSS地殻変動のみ

★ 検証すべきこと:
  gnss_score 単独で地震活動との相関があるか？
  もし gnss_score だけで有意な相関が出れば、それは循環ではない真のシグナル
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# データ読み込み
# ================================================================
OUTPUT_DIR = r"データ分析結果出力"

japan_scores = pd.read_csv(f"{OUTPUT_DIR}/japan_scores.csv", parse_dates=["date"])
japan_scores = japan_scores.rename(columns={"score": "japan_score"})
world_scores = pd.read_csv(f"{OUTPUT_DIR}/world_scores.csv", parse_dates=["date"])
depth_scores = pd.read_csv(f"{OUTPUT_DIR}/depth_scores.csv", parse_dates=["date"])
gnss_scores = pd.read_csv(f"{OUTPUT_DIR}/gnss_scores.csv", parse_dates=["date"])
b_value_scores = pd.read_csv(f"{OUTPUT_DIR}/b_value_scores.csv", parse_dates=["date"])
total = pd.read_csv(f"{OUTPUT_DIR}/total_with_gnss.csv", parse_dates=["date"])

eq_jp = pd.read_csv(r"生データ\2010-all.csv", parse_dates=["date"])
eq_world = pd.read_csv(r"生データ\world-all.csv", parse_dates=["date"])

# マージ
merged = total.copy()
merged = merged.merge(japan_scores, on="date", how="left")
merged = merged.merge(world_scores, on="date", how="left")
merged = merged.merge(depth_scores, on="date", how="left")
merged = merged.merge(gnss_scores, on="date", how="left")
merged = merged.merge(b_value_scores[["date", "b_value_score"]], on="date", how="left")
merged = merged.fillna(0)

# 地震データに依存するスコア vs 独立スコア
merged["circular_score"] = (merged["japan_score"] + merged["world_score"] +
                             merged["depth_score"] + merged["b_value_score"])
merged["independent_score"] = merged["gnss_score"]

print("=" * 80)
print("  循環論法テスト & 成分分離分析")
print("=" * 80)

# ================================================================
# 1. 各成分の基本統計
# ================================================================
print("\n■ 1. 各スコア成分の基本統計")
print("─" * 60)

components = {
    "japan_score": "国内地震スコア [循環]",
    "world_score": "世界地震スコア [循環]",
    "depth_score": "深さスコア [循環]",
    "b_value_score": "b値スコア [循環]",
    "gnss_score": "GNSSスコア [独立★]",
    "circular_score": "循環成分合計",
    "independent_score": "独立成分（GNSS）",
    "total_score": "総合スコア",
}

for col, label in components.items():
    vals = merged[col]
    print(f"  {label:<24} 平均={vals.mean():>7.3f}  SD={vals.std():>7.3f}  "
          f"最大={vals.max():>7.2f}  >0日数={len(vals[vals>0]):>5}")

# ================================================================
# 2. 各成分の寄与率
# ================================================================
print(f"\n■ 2. total_scoreへの寄与率")
print("─" * 60)

total_mean = merged["total_score"].mean()
for col in ["japan_score", "world_score", "depth_score", "b_value_score", "gnss_score"]:
    contrib = merged[col].mean() / total_mean * 100 if total_mean > 0 else 0
    print(f"  {components[col]:<24} 寄与率: {contrib:.1f}%")

# ================================================================
# 3. 核心テスト: 各成分と地震活動の相関（四分位分析）
# ================================================================
print(f"\n\n■ 3. 核心テスト: 各スコア成分 vs 地震活動（四分位分析）")
print("=" * 80)

# 日ごとの地震数を準備
all_dates = pd.date_range(merged["date"].min(), merged["date"].max(), freq="D")
merged["date_key"] = merged["date"].dt.date

datasets = {
    "日本M4+": eq_jp[eq_jp["magnitude"] >= 4.0],
    "日本M5+": eq_jp[eq_jp["magnitude"] >= 5.0],
    "世界M6+": eq_world[eq_world["magnitude_w"] >= 6.0],
    "世界M7+": eq_world[eq_world["magnitude_w"] >= 7.0],
}

for label, eqdata in datasets.items():
    if "magnitude_w" in eqdata.columns:
        daily = eqdata.groupby(eqdata["date"].dt.date).size()
    else:
        daily = eqdata.groupby(eqdata["date"].dt.date).size()
    merged[f"eq_{label}"] = merged["date_key"].map(daily).fillna(0)

# 各成分について四分位分析
test_scores = [
    ("gnss_score", "★ GNSSスコア [独立] ★"),
    ("japan_score", "国内地震スコア [循環]"),
    ("world_score", "世界地震スコア [循環]"),
    ("depth_score", "深さスコア [循環]"),
    ("b_value_score", "b値スコア [循環]"),
    ("circular_score", "循環成分合計"),
    ("independent_score", "独立成分（GNSS）"),
    ("total_score", "総合スコア"),
]

for score_col, score_label in test_scores:
    print(f"\n  ─── {score_label} ───")

    # スコアが全部同じ値だとqcutできないので確認
    if merged[score_col].nunique() < 4:
        # スコア=0と>0で2分割
        merged[f"{score_col}_grp"] = merged[score_col].apply(lambda x: "スコア>0" if x > 0 else "スコア=0")
        for eq_label in datasets.keys():
            eq_col = f"eq_{eq_label}"
            grp0 = merged[merged[f"{score_col}_grp"] == "スコア=0"][eq_col]
            grp1 = merged[merged[f"{score_col}_grp"] == "スコア>0"][eq_col]
            if len(grp1) > 0 and len(grp0) > 0:
                u_stat, mw_p = stats.mannwhitneyu(grp1, grp0, alternative='greater')
                print(f"    {eq_label}: スコア=0 平均={grp0.mean():.3f}(n={len(grp0)}) "
                      f"スコア>0 平均={grp1.mean():.3f}(n={len(grp1)}) "
                      f"MW-p={mw_p:.6f}{'**' if mw_p < 0.05 else ''}")
    else:
        try:
            merged[f"{score_col}_q"] = pd.qcut(merged[score_col], q=4,
                                                labels=["Q1(低)", "Q2", "Q3", "Q4(高)"],
                                                duplicates='drop')
        except ValueError:
            # 同値が多い場合はrank-based
            merged[f"{score_col}_q"] = pd.qcut(merged[score_col].rank(method='first'), q=4,
                                                labels=["Q1(低)", "Q2", "Q3", "Q4(高)"])

        for eq_label in datasets.keys():
            eq_col = f"eq_{eq_label}"
            q1 = merged[merged[f"{score_col}_q"] == "Q1(低)"][eq_col]
            q4 = merged[merged[f"{score_col}_q"] == "Q4(高)"][eq_col]

            u_stat, mw_p = stats.mannwhitneyu(q4, q1, alternative='greater')
            rho, sp_p = stats.spearmanr(merged[score_col], merged[eq_col])

            print(f"    {eq_label}: Q1平均={q1.mean():.3f} Q4平均={q4.mean():.3f} "
                  f"比率={q4.mean()/q1.mean():.2f}x MW-p={mw_p:.6f}{'**' if mw_p < 0.05 else ''} "
                  f"ρ={rho:+.4f}")


# ================================================================
# 4. 最重要: GNSS単独での「警報モデル」評価
# ================================================================
print(f"\n\n■ 4. GNSS単独での臨界状態検出")
print("=" * 80)

# GNSSスコアの分布を確認
gnss_vals = merged["gnss_score"]
print(f"  GNSSスコア分布:")
print(f"    0:     {len(gnss_vals[gnss_vals == 0])}日 ({len(gnss_vals[gnss_vals == 0])/len(gnss_vals)*100:.1f}%)")
print(f"    0<x<5: {len(gnss_vals[(gnss_vals > 0) & (gnss_vals < 5)])}日")
print(f"    5≤x<10: {len(gnss_vals[(gnss_vals >= 5) & (gnss_vals < 10)])}日")
print(f"    10≤x:  {len(gnss_vals[gnss_vals >= 10])}日")

# GNSSスコア > 0 の日の前後で地震活動を比較
gnss_active = merged[merged["gnss_score"] > 0].copy()
gnss_inactive = merged[merged["gnss_score"] == 0].copy()

print(f"\n  GNSS活動あり vs なし:")
for eq_label in datasets.keys():
    eq_col = f"eq_{eq_label}"
    active_mean = gnss_active[eq_col].mean()
    inactive_mean = gnss_inactive[eq_col].mean()
    ratio = active_mean / inactive_mean if inactive_mean > 0 else float('inf')
    u_stat, mw_p = stats.mannwhitneyu(gnss_active[eq_col], gnss_inactive[eq_col], alternative='greater')
    print(f"    {eq_label}: GNSS活動あり={active_mean:.3f} なし={inactive_mean:.3f} "
          f"比率={ratio:.2f}x MW-p={mw_p:.6f}{'**' if mw_p < 0.05 else ''}")

# ================================================================
# 5. 時間遅れ交差相関（GNSS → 地震活動）
# ================================================================
print(f"\n\n■ 5. 時間遅れ交差相関: GNSS変動 → 地震活動")
print("=" * 80)
print("  （GNSSスコアがN日後の地震活動と相関するかを調べる）")
print("  （正のラグ = GNSSが先行 = 予測的意味あり）")

for eq_label in ["日本M4+", "日本M5+", "世界M6+", "世界M7+"]:
    eq_col = f"eq_{eq_label}"
    gnss_series = merged["gnss_score"].values
    eq_series = merged[eq_col].values

    print(f"\n  【{eq_label}】")
    print(f"  {'ラグ(日)':>10} {'相関ρ':>10} {'p値':>12} {'有意':>5}")
    print(f"  " + "─" * 45)

    best_rho = 0
    best_lag = 0

    for lag in [0, 1, 3, 5, 7, 10, 14, 21, 30, 45, 60]:
        if lag == 0:
            g = gnss_series
            e = eq_series
        else:
            g = gnss_series[:-lag]
            e = eq_series[lag:]

        rho, p = stats.spearmanr(g, e)
        sig = "**" if p < 0.05 else ""
        print(f"  {lag:>10} {rho:>+10.4f} {p:>12.6f} {sig:>5}")

        if abs(rho) > abs(best_rho):
            best_rho = rho
            best_lag = lag

    print(f"  → 最大相関: ラグ={best_lag}日 ρ={best_rho:+.4f}")


# ================================================================
# 6. 結論
# ================================================================
print("\n\n" + "=" * 80)
print("  ■ 結論: 循環論法の影響評価")
print("=" * 80)
print("""
  total_score の5成分のうち4成分（japan_score, world_score, depth_score,
  b_value_score）は地震データそのものを入力としている。

  したがって、前回の分析で見つかった「スコアと地震活動の相関」は、
  大部分が循環論法（tautology）による見かけの相関である可能性が高い。

  しかし、唯一の独立成分である gnss_score について：
  - gnss_score は GNSS地殻変動データのみから計算される
  - gnss_score が地震活動と有意な相関を示すなら、
    それは地殻変動が地震活動を反映（または先行）する真のシグナル

  上記の分析結果を確認し、gnss_score 単独での有意性を評価すること。
""")
