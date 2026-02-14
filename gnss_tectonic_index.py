"""
Tectonic Stress Index (TSI) — GNSSベースの地殻状態指数
=====================================================
設計思想: 「大きく動いた日」ではなく「動き方が変わった期間」を検出

4成分:
  ① 速度変化 (Velocity Change): 7日MA - 30日MA の偏差
  ② 空間的整合性 (Spatial Coherence): 複数局の変位ベクトル方向一致率
  ③ 加速度 (Acceleration): 速度の時間微分
  ④ 低周波偏差 (Low-Frequency Anomaly): 30日MAからのトレンド逸脱
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# 1. データ読み込みと前処理
# ================================================================
print("=" * 80)
print("  Tectonic Stress Index (TSI) 構築")
print("=" * 80)

df = pd.read_csv(r"生データ\gnss_data.csv", parse_dates=["date"])
df = df.sort_values(["station_id", "date"]).reset_index(drop=True)

stations = sorted(df["station_id"].unique())
print(f"  観測点数: {len(stations)}")
print(f"  期間: {df['date'].min().date()} ～ {df['date'].max().date()}")

# ================================================================
# 2. 各局の速度・加速度・低周波偏差を計算
# ================================================================
print("\n■ 各局の時系列処理...")

all_station_daily = {}

for sid in stations:
    sdata = df[df["station_id"] == sid].copy()
    sdata = sdata.set_index("date").sort_index()
    name = sdata["station_name"].iloc[0]

    # 日次座標（E, N, U）
    coords = sdata[["east_mm", "north_mm", "up_mm"]].copy()

    # 欠損日を補間
    full_idx = pd.date_range(coords.index.min(), coords.index.max(), freq="D")
    coords = coords.reindex(full_idx)
    coords = coords.interpolate(method="linear", limit=7)  # 最大7日まで線形補間

    # --- ① 速度変化フィルタ: 7日MA - 30日MA ---
    ma7 = coords.rolling(7, center=True, min_periods=4).mean()
    ma30 = coords.rolling(30, center=True, min_periods=15).mean()
    velocity_change = ma7 - ma30  # トレンド変化検出フィルタ

    # 水平ベクトルの大きさ
    vc_horiz = np.sqrt(velocity_change["east_mm"]**2 + velocity_change["north_mm"]**2)
    # 3成分の大きさ
    vc_3d = np.sqrt(velocity_change["east_mm"]**2 +
                     velocity_change["north_mm"]**2 +
                     velocity_change["up_mm"]**2)

    # --- ② 速度（14日窓の傾き）---
    # 各成分の14日窓での線形回帰傾き（mm/day）
    velocity_e = coords["east_mm"].rolling(14, center=True, min_periods=7).apply(
        lambda x: np.polyfit(range(len(x)), x.values, 1)[0] if len(x.dropna()) >= 7 else np.nan, raw=False)
    velocity_n = coords["north_mm"].rolling(14, center=True, min_periods=7).apply(
        lambda x: np.polyfit(range(len(x)), x.values, 1)[0] if len(x.dropna()) >= 7 else np.nan, raw=False)
    velocity_horiz = np.sqrt(velocity_e**2 + velocity_n**2)

    # --- ③ 加速度（速度の7日変化率）---
    accel_e = velocity_e.diff(7) / 7
    accel_n = velocity_n.diff(7) / 7
    accel_horiz = np.sqrt(accel_e**2 + accel_n**2)

    # --- ④ 低周波偏差（30日MAの14日変化量）---
    lf_deviation = ma30.diff(14)
    lf_horiz = np.sqrt(lf_deviation["east_mm"]**2 + lf_deviation["north_mm"]**2)

    # 保存
    station_df = pd.DataFrame({
        "vc_east": velocity_change["east_mm"],
        "vc_north": velocity_change["north_mm"],
        "vc_up": velocity_change["up_mm"],
        "vc_horiz": vc_horiz,
        "vc_3d": vc_3d,
        "velocity_e": velocity_e,
        "velocity_n": velocity_n,
        "velocity_horiz": velocity_horiz,
        "accel_horiz": accel_horiz,
        "lf_horiz": lf_horiz,
    }, index=full_idx)
    station_df.index.name = "date"

    all_station_daily[sid] = station_df
    print(f"  {sid} ({name}): vc_horiz中央値={vc_horiz.median():.3f}mm "
          f"最大={vc_horiz.max():.3f}mm")


# ================================================================
# 3. 広域指標の統合（全局平均 + 空間整合性）
# ================================================================
print("\n■ 広域指標の統合...")

# 全日の日付リスト
all_dates = pd.date_range(df["date"].min() + timedelta(days=30),
                           df["date"].max() - timedelta(days=30), freq="D")

tsi_records = []

for date in all_dates:
    record = {"date": date}

    # --- 各局の値を収集 ---
    vc_horiz_vals = []
    vc_3d_vals = []
    accel_vals = []
    lf_vals = []

    # 空間整合性用: 各局の変位ベクトル（E, N）
    vectors_vc = []  # 速度変化ベクトル
    vectors_vel = []  # 速度ベクトル

    for sid in stations:
        sdf = all_station_daily[sid]
        if date not in sdf.index:
            continue
        row = sdf.loc[date]

        if not np.isnan(row["vc_horiz"]):
            vc_horiz_vals.append(row["vc_horiz"])
            vc_3d_vals.append(row["vc_3d"])
        if not np.isnan(row["accel_horiz"]):
            accel_vals.append(row["accel_horiz"])
        if not np.isnan(row["lf_horiz"]):
            lf_vals.append(row["lf_horiz"])

        # ベクトル方向
        if not np.isnan(row["vc_east"]) and not np.isnan(row["vc_north"]):
            vectors_vc.append([row["vc_east"], row["vc_north"]])
        if not np.isnan(row.get("velocity_e", np.nan)) and not np.isnan(row.get("velocity_n", np.nan)):
            vectors_vel.append([row["velocity_e"], row["velocity_n"]])

    # --- ① 速度変化指標（広域平均）---
    record["vc_horiz_mean"] = np.mean(vc_horiz_vals) if vc_horiz_vals else np.nan
    record["vc_horiz_max"] = np.max(vc_horiz_vals) if vc_horiz_vals else np.nan
    record["vc_3d_mean"] = np.mean(vc_3d_vals) if vc_3d_vals else np.nan

    # --- ② 空間的整合性（ベクトルの方向一致率）---
    coherence_vc = np.nan
    coherence_vel = np.nan

    if len(vectors_vc) >= 3:
        vecs = np.array(vectors_vc)
        # 各ベクトルを正規化
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1  # ゼロ除算回避
        unit_vecs = vecs / norms
        # 平均ベクトルの大きさ = 整合性指標（0～1）
        mean_vec = unit_vecs.mean(axis=0)
        coherence_vc = np.linalg.norm(mean_vec)

    if len(vectors_vel) >= 3:
        vecs = np.array(vectors_vel)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        unit_vecs = vecs / norms
        mean_vec = unit_vecs.mean(axis=0)
        coherence_vel = np.linalg.norm(mean_vec)

    record["coherence_vc"] = coherence_vc
    record["coherence_vel"] = coherence_vel

    # --- ③ 加速度指標 ---
    record["accel_mean"] = np.mean(accel_vals) if accel_vals else np.nan

    # --- ④ 低周波偏差 ---
    record["lf_mean"] = np.mean(lf_vals) if lf_vals else np.nan

    tsi_records.append(record)

tsi = pd.DataFrame(tsi_records)
tsi = tsi.set_index("date")

print(f"  TSIデータ: {len(tsi)}日")

# ================================================================
# 4. 各指標の標準化 & TSI合成
# ================================================================
print("\n■ TSI合成...")

# 各指標をZスコアに標準化
for col in ["vc_horiz_mean", "coherence_vc", "coherence_vel", "accel_mean", "lf_mean"]:
    mean = tsi[col].mean()
    std = tsi[col].std()
    tsi[f"{col}_z"] = (tsi[col] - mean) / std if std > 0 else 0

# TSI（各成分の重み付け和 — まず等重み）
tsi["TSI"] = (
    tsi["vc_horiz_mean_z"] +       # 速度変化
    tsi["coherence_vc_z"] +         # 空間整合性（速度変化ベクトル）
    tsi["accel_mean_z"] +           # 加速度
    tsi["lf_mean_z"]                # 低周波偏差
) / 4

# 30日移動平均TSI（ノイズ除去）
tsi["TSI_30d"] = tsi["TSI"].rolling(30, center=True, min_periods=15).mean()

print(f"  TSI統計: 平均={tsi['TSI'].mean():.4f} SD={tsi['TSI'].std():.4f}")
print(f"  TSI_30d統計: 平均={tsi['TSI_30d'].mean():.4f} SD={tsi['TSI_30d'].std():.4f}")

# ================================================================
# 5. TSI vs 地震活動の相関テスト
# ================================================================
print("\n\n" + "=" * 80)
print("  ■ TSI vs 地震活動の相関テスト")
print("=" * 80)

eq_jp = pd.read_csv(r"生データ\2010-all.csv", parse_dates=["date"])
eq_world = pd.read_csv(r"生データ\world-all.csv", parse_dates=["date"])

# 日ごとの地震数
all_eq_dates = pd.date_range(tsi.index.min(), tsi.index.max(), freq="D")
eq_daily = pd.DataFrame(index=all_eq_dates)

for label, eqdata, mcol, threshold in [
    ("日本M4+", eq_jp, "magnitude", 4.0),
    ("日本M5+", eq_jp, "magnitude", 5.0),
    ("日本M6+", eq_jp, "magnitude", 6.0),
    ("世界M6+", eq_world, "magnitude_w", 6.0),
    ("世界M6.5+", eq_world, "magnitude_w", 6.5),
    ("世界M7+", eq_world, "magnitude_w", 7.0),
]:
    subset = eqdata[eqdata[mcol] >= threshold]
    daily = subset.groupby(subset["date"].dt.date).size()
    eq_daily[label] = pd.Series(0, index=all_eq_dates.date)
    for d, cnt in daily.items():
        if d in eq_daily.index.date:
            eq_daily.loc[eq_daily.index.date == d, label] = cnt
    eq_daily[label] = eq_daily[label].fillna(0).astype(float)

# TSIとマージ
tsi_eq = tsi.join(eq_daily, how="inner")

# --- 5a. 四分位分析 ---
print("\n  ─── 四分位分析 ───")

test_indicators = [
    ("TSI", "TSI（日次）"),
    ("TSI_30d", "TSI（30日MA）"),
    ("vc_horiz_mean", "① 速度変化（水平）"),
    ("coherence_vc", "② 空間整合性（速度変化）"),
    ("coherence_vel", "② 空間整合性（速度）"),
    ("accel_mean", "③ 加速度"),
    ("lf_mean", "④ 低周波偏差"),
]

eq_labels = ["日本M4+", "日本M5+", "世界M6+", "世界M7+"]

for ind_col, ind_label in test_indicators:
    print(f"\n  【{ind_label}】")
    valid = tsi_eq[ind_col].dropna()
    if len(valid) < 100:
        print(f"    データ不足 ({len(valid)}件)")
        continue

    try:
        tsi_eq[f"{ind_col}_q"] = pd.qcut(tsi_eq[ind_col].rank(method="first"), q=4,
                                           labels=["Q1(低)", "Q2", "Q3", "Q4(高)"])
    except:
        print(f"    四分位分割不可")
        continue

    for eq_label in eq_labels:
        q1 = tsi_eq[tsi_eq[f"{ind_col}_q"] == "Q1(低)"][eq_label]
        q4 = tsi_eq[tsi_eq[f"{ind_col}_q"] == "Q4(高)"][eq_label]

        ratio = q4.mean() / q1.mean() if q1.mean() > 0 else float('inf')
        u_stat, mw_p = stats.mannwhitneyu(q4, q1, alternative='greater')
        rho, sp_p = stats.spearmanr(tsi_eq[ind_col].dropna(),
                                      tsi_eq.loc[tsi_eq[ind_col].notna(), eq_label])

        sig = "**" if mw_p < 0.05 else ""
        print(f"    {eq_label}: Q1={q1.mean():.3f} Q4={q4.mean():.3f} "
              f"比率={ratio:.2f}x MW-p={mw_p:.6f}{sig} ρ={rho:+.4f}")


# --- 5b. 時間遅れ交差相関 ---
print(f"\n\n  ─── 時間遅れ交差相関: TSI → 地震活動 ───")
print(f"  （正のラグ = TSIが先行 = 予測的意味あり）")

for tsi_col, tsi_label in [("TSI", "TSI日次"), ("TSI_30d", "TSI 30日MA")]:
    print(f"\n  【{tsi_label}】")
    tsi_vals = tsi_eq[tsi_col].values

    for eq_label in eq_labels:
        eq_vals = tsi_eq[eq_label].values
        best_rho = 0
        best_lag = 0
        results_str = []

        for lag in [0, 3, 7, 14, 21, 30, 45, 60]:
            if lag == 0:
                g = tsi_vals
                e = eq_vals
            else:
                g = tsi_vals[:-lag]
                e = eq_vals[lag:]

            valid_mask = ~(np.isnan(g) | np.isnan(e))
            if valid_mask.sum() < 100:
                continue

            rho, p = stats.spearmanr(g[valid_mask], e[valid_mask])
            sig = "**" if p < 0.05 else ""

            if abs(rho) > abs(best_rho):
                best_rho = rho
                best_lag = lag
            results_str.append(f"lag{lag}d:ρ={rho:+.4f}{sig}")

        print(f"    {eq_label}: {' | '.join(results_str)}")
        print(f"      → 最大: lag={best_lag}d ρ={best_rho:+.4f}")


# --- 5c. 各成分の相関（30日MA版） ---
print(f"\n\n  ─── 各成分の30日MA版での相関 ───")
for col in ["vc_horiz_mean", "coherence_vc", "accel_mean", "lf_mean"]:
    tsi_eq[f"{col}_30d"] = tsi_eq[col].rolling(30, center=True, min_periods=15).mean()

for ind_col_base in ["vc_horiz_mean", "coherence_vc", "accel_mean", "lf_mean"]:
    ind_col = f"{ind_col_base}_30d"
    ind_label = {
        "vc_horiz_mean": "① 速度変化",
        "coherence_vc": "② 空間整合性",
        "accel_mean": "③ 加速度",
        "lf_mean": "④ 低周波偏差",
    }[ind_col_base] + "（30日MA）"

    print(f"\n  【{ind_label}】")
    for eq_label in eq_labels:
        valid = tsi_eq[[ind_col, eq_label]].dropna()
        if len(valid) < 100:
            continue

        # 30日MA地震数
        eq_30d = tsi_eq[eq_label].rolling(30, center=True, min_periods=15).mean()
        valid2 = pd.DataFrame({
            "ind": tsi_eq[ind_col],
            "eq": eq_30d
        }).dropna()

        if len(valid2) < 100:
            continue

        rho, p = stats.spearmanr(valid2["ind"], valid2["eq"])
        sig = "**" if p < 0.05 else ""
        print(f"    {eq_label}: ρ={rho:+.4f} p={p:.6f}{sig} (n={len(valid2)})")


# ================================================================
# 6. 保存
# ================================================================
output_cols = ["vc_horiz_mean", "vc_3d_mean", "coherence_vc", "coherence_vel",
               "accel_mean", "lf_mean", "TSI", "TSI_30d"]
tsi[output_cols].to_csv(r"データ分析結果出力\tectonic_stress_index.csv")
print(f"\n\n保存完了: データ分析結果出力/tectonic_stress_index.csv")


# ================================================================
# 7. TSI高値期間の詳細
# ================================================================
print("\n\n" + "=" * 80)
print("  ■ TSI高値期間（上位5%）の詳細")
print("=" * 80)

tsi_95 = tsi["TSI_30d"].quantile(0.95)
tsi_high = tsi[tsi["TSI_30d"] >= tsi_95].copy()
print(f"  TSI_30d 95%ile閾値: {tsi_95:.4f}")
print(f"  高値日数: {len(tsi_high)}")

# クラスタリング
if len(tsi_high) > 0:
    high_dates = sorted(tsi_high.index.tolist())
    clusters = []
    cs = ce = high_dates[0]
    for d in high_dates[1:]:
        if (d - ce).days <= 5:
            ce = d
        else:
            clusters.append((cs, ce))
            cs = ce = d
    clusters.append((cs, ce))

    print(f"  クラスタ数: {len(clusters)}")
    print()

    m7_jp = eq_jp[eq_jp["magnitude"] >= 7.0]
    m7_w = eq_world[eq_world["magnitude_w"] >= 7.0]

    for i, (cs, ce) in enumerate(clusters):
        duration = (ce - cs).days + 1
        peak_val = tsi.loc[cs:ce, "TSI_30d"].max()
        peak_date = tsi.loc[cs:ce, "TSI_30d"].idxmax()

        # 60日以内のM7
        check_end = ce + timedelta(days=60)
        m7_jp_match = m7_jp[(m7_jp["date"] >= cs) & (m7_jp["date"] <= check_end)]
        m7_w_match = m7_w[(m7_w["date"] >= cs) & (m7_w["date"] <= check_end)]

        print(f"  クラスタ{i+1}: {cs.date()} ～ {ce.date()} ({duration}日間) "
              f"TSI_30dピーク={peak_val:.4f} ({peak_date.date()})")

        if len(m7_jp_match) > 0:
            for _, r in m7_jp_match.iterrows():
                days_after = (r["date"] - peak_date).days
                print(f"    → 日本M7: {r['date'].date()} {r['place']} M{r['magnitude']} "
                      f"(ピークから{days_after:+d}日)")
        if len(m7_w_match) > 0:
            for _, r in m7_w_match.iterrows():
                days_after = (r["date"] - peak_date).days
                print(f"    → 世界M7: {r['date'].date()} {r['place_w']} M{r['magnitude_w']} "
                      f"(ピークから{days_after:+d}日)")
        if len(m7_jp_match) == 0 and len(m7_w_match) == 0:
            print(f"    → 60日以内にM7なし")
