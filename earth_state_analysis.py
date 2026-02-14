"""
地殻状態モニタリング分析 (Earth State Monitoring)
=================================================
目的: スコアが「地震予知」ではなく「地球の臨界状態」を反映しているか検証

(A) 時間的偏り: スコア高値期間に地震活動そのものが増えるか
(B) 空間分布: スコア高値時にどの地域で地震が増えるか
(C) 先行時間分布: スコアピーク→大地震までの日数の分布
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
score = pd.read_csv(r"データ分析結果出力\total_with_gnss.csv", parse_dates=["date"])
eq_jp = pd.read_csv(r"生データ\2010-all.csv", parse_dates=["date"])
eq_world = pd.read_csv(r"生データ\world-all.csv", parse_dates=["date"])

score = score.sort_values("date").reset_index(drop=True)
eq_jp = eq_jp.sort_values("date").reset_index(drop=True)
eq_world = eq_world.sort_values("date").reset_index(drop=True)

THRESHOLD = 18
WINDOW = 30

print("=" * 80)
print("  地殻状態モニタリング分析 (Earth State Monitoring)")
print("=" * 80)

# ================================================================
# (A) 時間的偏り分析 — 最重要
# ================================================================
print("\n" + "=" * 80)
print("  (A) 時間的偏り分析: スコア高値期間に地震活動は増えるか？")
print("=" * 80)

# --- 警報クラスタの構築 ---
alerts = score[score["total_score"] >= THRESHOLD].copy()
alert_dates = sorted(alerts["date"].tolist())

clusters = []
if alert_dates:
    cs, ce = alert_dates[0], alert_dates[0]
    cmax = alerts[alerts["date"] == alert_dates[0]]["total_score"].values[0]
    cpeak_date = alert_dates[0]
    for d in alert_dates[1:]:
        if (d - ce).days <= 2:
            ce = d
            s = alerts[alerts["date"] == d]["total_score"].values[0]
            if s > cmax:
                cmax = s
                cpeak_date = d
        else:
            clusters.append({"start": cs, "end": ce, "peak_date": cpeak_date, "max_score": cmax})
            cs, ce = d, d
            cmax = alerts[alerts["date"] == d]["total_score"].values[0]
            cpeak_date = d
    clusters.append({"start": cs, "end": ce, "peak_date": cpeak_date, "max_score": cmax})

print(f"\n  警報クラスタ数: {len(clusters)}")

# --- 方法1: 警報日±30日 vs 非警報期間の地震数比較 ---
print("\n  ─── 方法1: 警報ウィンドウ内 vs 外の日平均地震数 ───")

# 警報ウィンドウの日セット構築
alert_window_dates = set()
for c in clusters:
    for i in range(-(WINDOW), (c["end"] - c["start"]).days + WINDOW + 1):
        d = (c["start"] + timedelta(days=i)).date()
        alert_window_dates.add(d)

# 日ごとの地震数を集計
all_dates = pd.date_range(score["date"].min(), score["date"].max(), freq="D")

datasets = {
    "日本 M4+": eq_jp[eq_jp["magnitude"] >= 4.0],
    "日本 M5+": eq_jp[eq_jp["magnitude"] >= 5.0],
    "日本 M6+": eq_jp[eq_jp["magnitude"] >= 6.0],
    "世界 M6+": eq_world[eq_world["magnitude_w"] >= 6.0],
    "世界 M6.5+": eq_world[eq_world["magnitude_w"] >= 6.5],
    "世界 M7+": eq_world[eq_world["magnitude_w"] >= 7.0],
}

print(f"\n  {'カテゴリ':<14} {'警報期間平均':>10} {'通常期間平均':>10} {'比率':>7} {'t検定p値':>10} {'Mann-W p値':>10}")
print("  " + "─" * 70)

for label, eqdata in datasets.items():
    # 日ごとの地震数
    if "magnitude_w" in eqdata.columns:
        daily = eqdata.groupby(eqdata["date"].dt.date).size()
    else:
        daily = eqdata.groupby(eqdata["date"].dt.date).size()

    # 全日に展開（0埋め）
    daily_full = pd.Series(0, index=[d.date() for d in all_dates])
    for d, cnt in daily.items():
        if d in daily_full.index:
            daily_full[d] = cnt

    # 警報期間 vs 非警報期間
    alert_counts = [daily_full[d] for d in daily_full.index if d in alert_window_dates]
    normal_counts = [daily_full[d] for d in daily_full.index if d not in alert_window_dates]

    mean_alert = np.mean(alert_counts)
    mean_normal = np.mean(normal_counts)
    ratio = mean_alert / mean_normal if mean_normal > 0 else float('inf')

    # Welch's t-test
    t_stat, t_p = stats.ttest_ind(alert_counts, normal_counts, equal_var=False)
    # Mann-Whitney U
    u_stat, mw_p = stats.mannwhitneyu(alert_counts, normal_counts, alternative='greater')

    sig_t = "**" if t_p < 0.05 else ""
    sig_mw = "**" if mw_p < 0.05 else ""

    print(f"  {label:<14} {mean_alert:>10.3f} {mean_normal:>10.3f} {ratio:>6.2f}x {t_p:>9.6f}{sig_t} {mw_p:>9.6f}{sig_mw}")

# --- 方法2: 前後対称ウィンドウ（より厳密） ---
print(f"\n\n  ─── 方法2: 各クラスタのピーク前後±30日 vs 対照期間（ピーク±90～120日） ───")

for label, eqdata in datasets.items():
    if "magnitude_w" in eqdata.columns:
        daily = eqdata.groupby(eqdata["date"].dt.date).size()
    else:
        daily = eqdata.groupby(eqdata["date"].dt.date).size()

    daily_full = pd.Series(0, index=[d.date() for d in all_dates])
    for d, cnt in daily.items():
        if d in daily_full.index:
            daily_full[d] = cnt

    near_counts = []  # ピーク±30日
    far_counts = []   # ピーク±90～120日（対照）

    for c in clusters:
        peak = c["peak_date"].date()
        for offset in range(-WINDOW, WINDOW + 1):
            d = peak + timedelta(days=offset)
            if d in daily_full.index:
                near_counts.append(daily_full[d])
        # 対照期間
        for offset in list(range(-120, -89)) + list(range(90, 121)):
            d = peak + timedelta(days=offset)
            if d in daily_full.index:
                far_counts.append(daily_full[d])

    if len(near_counts) > 0 and len(far_counts) > 0:
        mean_near = np.mean(near_counts)
        mean_far = np.mean(far_counts)
        ratio = mean_near / mean_far if mean_far > 0 else float('inf')
        t_stat, t_p = stats.ttest_ind(near_counts, far_counts, equal_var=False)
        u_stat, mw_p = stats.mannwhitneyu(near_counts, far_counts, alternative='greater')
        sig_t = "**" if t_p < 0.05 else ""
        sig_mw = "**" if mw_p < 0.05 else ""
        print(f"  {label:<14} 近傍={mean_near:.3f} 対照={mean_far:.3f} "
              f"比率={ratio:.2f}x t検定p={t_p:.6f}{sig_t} MWp={mw_p:.6f}{sig_mw}")

# --- 方法3: スコア四分位別の地震数（連続的な関係） ---
print(f"\n\n  ─── 方法3: スコア四分位別の日平均地震数 ───")

# スコアに日ごとの地震数を結合
score_daily = score.copy()
score_daily["date_key"] = score_daily["date"].dt.date

for label, eqdata in datasets.items():
    if "magnitude_w" in eqdata.columns:
        daily = eqdata.groupby(eqdata["date"].dt.date).size().rename("count")
    else:
        daily = eqdata.groupby(eqdata["date"].dt.date).size().rename("count")

    score_daily[f"eq_{label}"] = score_daily["date_key"].map(daily).fillna(0)

# 四分位
score_daily["score_quartile"] = pd.qcut(
    score_daily["total_score"], q=4, labels=["Q1(低)", "Q2", "Q3", "Q4(高)"]
)

# さらにtop 5%も
score_95 = score_daily["total_score"].quantile(0.95)
score_daily["score_top5"] = score_daily["total_score"] >= score_95

print(f"  スコア四分位の閾値:")
for q in [0.25, 0.50, 0.75, 0.95]:
    print(f"    {q*100:.0f}%ile: {score_daily['total_score'].quantile(q):.2f}")
print()

for label in datasets.keys():
    col = f"eq_{label}"
    print(f"  【{label}】")
    grouped = score_daily.groupby("score_quartile")[col].agg(["mean", "std", "count"])
    for q_label, row in grouped.iterrows():
        print(f"    {q_label}: 平均={row['mean']:.3f} (SD={row['std']:.3f}, n={int(row['count'])})")

    # Q1 vs Q4の検定
    q1_vals = score_daily[score_daily["score_quartile"] == "Q1(低)"][col]
    q4_vals = score_daily[score_daily["score_quartile"] == "Q4(高)"][col]
    t_stat, t_p = stats.ttest_ind(q1_vals, q4_vals, equal_var=False)
    u_stat, mw_p = stats.mannwhitneyu(q4_vals, q1_vals, alternative='greater')

    # Top 5%
    top5_mean = score_daily[score_daily["score_top5"]][col].mean()
    rest_mean = score_daily[~score_daily["score_top5"]][col].mean()

    print(f"    Q1 vs Q4: t検定p={t_p:.6f} MW-p={mw_p:.6f}")
    print(f"    Top 5%(≥{score_95:.2f}): 平均={top5_mean:.3f} vs 他={rest_mean:.3f} "
          f"比率={top5_mean/rest_mean:.2f}x" if rest_mean > 0 else "")
    print()

# --- 方法4: スピアマン相関（スコアと地震数の単調相関） ---
print(f"\n  ─── 方法4: スコアと地震数のスピアマン順位相関 ───")
print(f"  （日次、7日移動平均、30日移動平均）")
print()

for label in datasets.keys():
    col = f"eq_{label}"

    # 日次
    rho_d, p_d = stats.spearmanr(score_daily["total_score"], score_daily[col])

    # 7日移動平均
    score_7d = score_daily["total_score"].rolling(7, center=True).mean().dropna()
    eq_7d = score_daily[col].rolling(7, center=True).mean().dropna()
    idx = score_7d.index.intersection(eq_7d.index)
    rho_7, p_7 = stats.spearmanr(score_7d[idx], eq_7d[idx])

    # 30日移動平均
    score_30d = score_daily["total_score"].rolling(30, center=True).mean().dropna()
    eq_30d = score_daily[col].rolling(30, center=True).mean().dropna()
    idx30 = score_30d.index.intersection(eq_30d.index)
    rho_30, p_30 = stats.spearmanr(score_30d[idx30], eq_30d[idx30])

    sig_d = "**" if p_d < 0.05 else ""
    sig_7 = "**" if p_7 < 0.05 else ""
    sig_30 = "**" if p_30 < 0.05 else ""

    print(f"  {label:<14} 日次: ρ={rho_d:+.4f} p={p_d:.6f}{sig_d} | "
          f"7日: ρ={rho_7:+.4f} p={p_7:.6f}{sig_7} | "
          f"30日: ρ={rho_30:+.4f} p={p_30:.6f}{sig_30}")


# ================================================================
# (B) 空間分布分析
# ================================================================
print("\n\n" + "=" * 80)
print("  (B) 空間分布分析: スコア高値時にどの地域で地震が増えるか？")
print("=" * 80)

# --- 世界データの地域分類 ---
region_map = {
    "Japan": "日本周辺",
    "Japan region": "日本周辺",
    "Russia": "カムチャツカ・千島",
    "Kuril": "カムチャツカ・千島",
    "Kamchatka": "カムチャツカ・千島",
    "Indonesia": "インドネシア",
    "Philippines": "フィリピン",
    "Taiwan": "台湾",
    "Papua New Guinea": "パプアニューギニア",
    "Tonga": "トンガ・ケルマディック",
    "Kermadec": "トンガ・ケルマディック",
    "Fiji": "フィジー",
    "New Zealand": "ニュージーランド",
    "Chile": "南米西岸",
    "Peru": "南米西岸",
    "Ecuador": "南米西岸",
    "Colombia": "南米西岸",
    "Argentina": "南米西岸",
    "Mexico": "中米",
    "Guatemala": "中米",
    "Costa Rica": "中米",
    "El Salvador": "中米",
    "Nicaragua": "中米",
    "Honduras": "中米",
    "Panama": "中米",
    "United States": "アラスカ・北米西岸",
    "Alaska": "アラスカ・北米西岸",
    "Canada": "アラスカ・北米西岸",
    "China": "中国",
    "India": "南アジア",
    "Nepal": "南アジア",
    "Afghanistan": "南アジア",
    "Pakistan": "南アジア",
    "Iran": "中東",
    "Turkey": "中東",
    "Greece": "地中海",
    "Italy": "地中海",
    "Solomon": "ソロモン・バヌアツ",
    "Vanuatu": "ソロモン・バヌアツ",
    "South Sandwich": "南大西洋",
    "Sandwich": "南大西洋",
}

def classify_region(place):
    if pd.isna(place):
        return "その他"
    for key, region in region_map.items():
        if key.lower() in str(place).lower():
            return region
    return "その他"

eq_world["region"] = eq_world["place_w"].apply(classify_region)

# 日本データの地域分類
jp_region_map = {
    "三陸": "日本海溝（東北）",
    "宮城": "日本海溝（東北）",
    "岩手": "日本海溝（東北）",
    "福島": "日本海溝（東北）",
    "茨城": "日本海溝（関東）",
    "千葉": "日本海溝（関東）",
    "関東": "日本海溝（関東）",
    "北海道": "千島海溝",
    "釧路": "千島海溝",
    "根室": "千島海溝",
    "十勝": "千島海溝",
    "択捉": "千島海溝",
    "国後": "千島海溝",
    "石川": "日本海側",
    "新潟": "日本海側",
    "秋田": "日本海側",
    "山形": "日本海側",
    "能登": "日本海側",
    "南海": "南海トラフ",
    "東南海": "南海トラフ",
    "紀伊": "南海トラフ",
    "和歌山": "南海トラフ",
    "日向灘": "南海トラフ",
    "熊本": "九州",
    "鹿児島": "九州",
    "大分": "九州",
    "宮崎": "九州",
    "薩摩": "九州",
    "トカラ": "九州・南西諸島",
    "沖縄": "九州・南西諸島",
    "奄美": "九州・南西諸島",
    "小笠原": "伊豆・小笠原",
    "父島": "伊豆・小笠原",
    "鳥島": "伊豆・小笠原",
    "伊豆": "伊豆・小笠原",
    "カムチャツカ": "カムチャツカ",
    "台湾": "台湾付近",
}

def classify_jp_region(place):
    if pd.isna(place):
        return "その他"
    for key, region in jp_region_map.items():
        if key in str(place):
            return region
    return "その他"

eq_jp["region"] = eq_jp["place"].apply(classify_jp_region)

# --- 警報期間 vs 非警報期間の地域別地震数 ---
print(f"\n  ─── 世界M6+: 警報±30日 vs 通常期間 の地域別比較 ───")

eq_world["date_key"] = eq_world["date"].dt.date
eq_world["in_alert_window"] = eq_world["date_key"].apply(lambda d: d in alert_window_dates)

n_alert_days = len(alert_window_dates)
n_normal_days = len(set(d.date() for d in all_dates) - alert_window_dates)

print(f"  警報ウィンドウ日数: {n_alert_days}日 / 通常日数: {n_normal_days}日")
print()

regions_world = eq_world["region"].value_counts()
print(f"  {'地域':<22} {'警報期間':>6} {'通常期間':>6} {'警報日平均':>10} {'通常日平均':>10} {'比率':>7}")
print("  " + "─" * 70)

region_ratios_world = []
for region in regions_world.index:
    subset = eq_world[eq_world["region"] == region]
    n_alert = subset[subset["in_alert_window"]].shape[0]
    n_normal = subset[~subset["in_alert_window"]].shape[0]
    rate_alert = n_alert / n_alert_days if n_alert_days > 0 else 0
    rate_normal = n_normal / n_normal_days if n_normal_days > 0 else 0
    ratio = rate_alert / rate_normal if rate_normal > 0 else float('inf')
    region_ratios_world.append((region, n_alert, n_normal, rate_alert, rate_normal, ratio))
    marker = " ★" if ratio >= 1.5 else (" ▲" if ratio >= 1.2 else "")
    print(f"  {region:<22} {n_alert:>6} {n_normal:>6} {rate_alert:>10.4f} {rate_normal:>10.4f} {ratio:>6.2f}x{marker}")

# --- 日本M4+の地域別 ---
print(f"\n\n  ─── 日本M4+: 警報±30日 vs 通常期間 の地域別比較 ───")

eq_jp_m4 = eq_jp[eq_jp["magnitude"] >= 4.0].copy()
eq_jp_m4["date_key"] = eq_jp_m4["date"].dt.date
eq_jp_m4["in_alert_window"] = eq_jp_m4["date_key"].apply(lambda d: d in alert_window_dates)

regions_jp = eq_jp_m4["region"].value_counts()
print(f"  {'地域':<22} {'警報期間':>6} {'通常期間':>6} {'警報日平均':>10} {'通常日平均':>10} {'比率':>7}")
print("  " + "─" * 70)

for region in regions_jp.index:
    if regions_jp[region] < 20:
        continue
    subset = eq_jp_m4[eq_jp_m4["region"] == region]
    n_alert = subset[subset["in_alert_window"]].shape[0]
    n_normal = subset[~subset["in_alert_window"]].shape[0]
    rate_alert = n_alert / n_alert_days if n_alert_days > 0 else 0
    rate_normal = n_normal / n_normal_days if n_normal_days > 0 else 0
    ratio = rate_alert / rate_normal if rate_normal > 0 else float('inf')
    marker = " ★" if ratio >= 1.5 else (" ▲" if ratio >= 1.2 else "")
    print(f"  {region:<22} {n_alert:>6} {n_normal:>6} {rate_alert:>10.4f} {rate_normal:>10.4f} {ratio:>6.2f}x{marker}")


# ================================================================
# (C) 先行時間（リードタイム）分布
# ================================================================
print("\n\n" + "=" * 80)
print("  (C) 先行時間分布: スコアピーク → 大地震までの日数")
print("=" * 80)

# 各クラスタピークから最寄りのM7までの日数
m7_jp = eq_jp[eq_jp["magnitude"] >= 7.0].copy()
m7_world = eq_world[eq_world["magnitude_w"] >= 7.0].copy()

print(f"\n  ─── 各クラスタピーク → 最寄りM7（90日以内、世界データ） ───")
lead_times = []
for c in clusters:
    peak = c["peak_date"]
    # 90日以内のM7（世界）
    future_m7 = m7_world[(m7_world["date"] > peak) &
                          (m7_world["date"] <= peak + timedelta(days=90))]
    if len(future_m7) > 0:
        nearest = future_m7.iloc[0]
        days = (nearest["date"] - peak).days
        lead_times.append(days)
        print(f"  ピーク {peak.date()} (スコア={c['max_score']:.2f}) "
              f"→ {days}日後 {nearest['place_w']} M{nearest['magnitude_w']}")
    else:
        print(f"  ピーク {peak.date()} (スコア={c['max_score']:.2f}) → 90日以内にM7なし")

if lead_times:
    print(f"\n  リードタイム統計:")
    print(f"    件数: {len(lead_times)}")
    print(f"    平均: {np.mean(lead_times):.1f}日")
    print(f"    中央値: {np.median(lead_times):.1f}日")
    print(f"    範囲: {min(lead_times)}～{max(lead_times)}日")
    print(f"    分布: {sorted(lead_times)}")

# 日本M7も
print(f"\n  ─── 各クラスタピーク → 最寄りM7（90日以内、日本データ） ───")
lead_times_jp = []
for c in clusters:
    peak = c["peak_date"]
    future_m7_jp = m7_jp[(m7_jp["date"] > peak) &
                          (m7_jp["date"] <= peak + timedelta(days=90))]
    # 当日も含める
    same_day = m7_jp[m7_jp["date"] == peak]
    combined = pd.concat([same_day, future_m7_jp]).drop_duplicates()

    if len(combined) > 0:
        nearest = combined.iloc[0]
        days = (nearest["date"] - peak).days
        lead_times_jp.append(days)
        print(f"  ピーク {peak.date()} (スコア={c['max_score']:.2f}) "
              f"→ {days}日後 {nearest['place']} M{nearest['magnitude']}")
    else:
        print(f"  ピーク {peak.date()} (スコア={c['max_score']:.2f}) → 90日以内にM7なし（日本）")


# ================================================================
# 追加: ランダム化検定（地震活動量の差）
# ================================================================
print("\n\n" + "=" * 80)
print("  (D) ランダム化検定: スコア高値期間の地震活動増加は偶然か？")
print("=" * 80)

np.random.seed(42)
N_PERM = 10_000

# 日本M4+の日次地震数
daily_jp_m4 = eq_jp[eq_jp["magnitude"] >= 4.0].groupby(eq_jp[eq_jp["magnitude"] >= 4.0]["date"].dt.date).size()
daily_jp_m4_full = pd.Series(0.0, index=[d.date() for d in all_dates])
for d, cnt in daily_jp_m4.items():
    if d in daily_jp_m4_full.index:
        daily_jp_m4_full[d] = cnt

# 世界M6+の日次地震数
daily_w_m6 = eq_world.groupby(eq_world["date"].dt.date).size()
daily_w_m6_full = pd.Series(0.0, index=[d.date() for d in all_dates])
for d, cnt in daily_w_m6.items():
    if d in daily_w_m6_full.index:
        daily_w_m6_full[d] = cnt

# 観測された警報期間の平均地震数
alert_dates_set = alert_window_dates
obs_mean_jp = np.mean([daily_jp_m4_full[d] for d in daily_jp_m4_full.index if d in alert_dates_set])
obs_mean_w = np.mean([daily_w_m6_full[d] for d in daily_w_m6_full.index if d in alert_dates_set])

# ランダムに同数の日を選んで平均を計算（パーミュテーション検定）
all_dates_list = list(daily_jp_m4_full.index)
n_alert_total = len(alert_dates_set)

perm_means_jp = np.zeros(N_PERM)
perm_means_w = np.zeros(N_PERM)

jp_values = daily_jp_m4_full.values
w_values = daily_w_m6_full.values
n_total = len(all_dates_list)

for i in range(N_PERM):
    idx = np.random.choice(n_total, size=n_alert_total, replace=False)
    perm_means_jp[i] = jp_values[idx].mean()
    perm_means_w[i] = w_values[idx].mean()

p_perm_jp = np.mean(perm_means_jp >= obs_mean_jp)
p_perm_w = np.mean(perm_means_w >= obs_mean_w)

print(f"\n  パーミュテーション検定（{N_PERM:,}回）")
print(f"  ─────────────────────────────────────────")
print(f"  日本M4+: 観測平均={obs_mean_jp:.3f} ランダム平均={np.mean(perm_means_jp):.3f} p値={p_perm_jp:.6f}")
print(f"  世界M6+: 観測平均={obs_mean_w:.3f} ランダム平均={np.mean(perm_means_w):.3f} p値={p_perm_w:.6f}")

if p_perm_jp < 0.05 or p_perm_w < 0.05:
    print(f"\n  → 偶然では説明しにくい地震活動の増加が検出された")
else:
    print(f"\n  → 現時点では偶然の範囲内（ただしサンプル不足の可能性あり）")


# ================================================================
# 総合まとめ
# ================================================================
print("\n\n" + "=" * 80)
print("  総合まとめ")
print("=" * 80)
print("""
  本分析では、スコアが「地震予知」ではなく「地殻の臨界状態」を
  反映しているかを3つの観点から検証した。

  (A) 時間的偏り:
      スコア高値期間に地震活動が統計的に増加するかを、
      4つの方法（ウィンドウ比較、対照期間比較、四分位分析、相関分析）で評価。

  (B) 空間分布:
      スコア高値時にどの地域で地震が増えるかを地域別に評価。
      特定のプレート境界でのみ増加が見られれば、物理的意味がある。

  (C) 先行時間:
      スコアピークから大地震までの日数分布を確認。
      特定の範囲に集中すれば、物理過程を反映している可能性がある。

  (D) ランダム化検定:
      パーミュテーション検定で偶然を排除。
""")
