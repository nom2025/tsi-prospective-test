"""
警報モデルの統計的有意性検定
- 二項検定（的中率がベースレートより有意に高いか）
- フィッシャーの正確確率検定（2×2分割表）
- モンテカルロシミュレーション（独立警報イベント数を考慮）
- 連続警報日のクラスタリング（独立イベント評価）
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy import stats

# --- データ読み込み ---
score = pd.read_csv(r"データ分析結果出力\total_with_gnss.csv", parse_dates=["date"])
eq = pd.read_csv(r"生データ\2010-all.csv", parse_dates=["date"])
score = score.sort_values("date").reset_index(drop=True)
eq = eq.sort_values("date").reset_index(drop=True)
m7 = eq[eq["magnitude"] >= 7.0].copy()

THRESHOLD = 18
WINDOW = 30

alerts = score[score["total_score"] >= THRESHOLD].copy()
total_days = (score["date"].max() - score["date"].min()).days + 1

print("=" * 70)
print("  警報モデル 統計的有意性検定")
print("=" * 70)

# ================================================================
# 1. 連続警報日のクラスタリング（独立イベント評価）
# ================================================================
print("\n■ 1. 警報クラスタリング（連続日を1イベントとみなす）")

alert_dates = sorted(alerts["date"].tolist())
clusters = []
if alert_dates:
    cluster_start = alert_dates[0]
    cluster_end = alert_dates[0]
    cluster_max_score = alerts[alerts["date"] == alert_dates[0]]["total_score"].values[0]

    for d in alert_dates[1:]:
        if (d - cluster_end).days <= 2:  # 2日以内は同一クラスタ
            cluster_end = d
            s = alerts[alerts["date"] == d]["total_score"].values[0]
            cluster_max_score = max(cluster_max_score, s)
        else:
            clusters.append({
                "start": cluster_start,
                "end": cluster_end,
                "days": (cluster_end - cluster_start).days + 1,
                "max_score": cluster_max_score
            })
            cluster_start = d
            cluster_end = d
            cluster_max_score = alerts[alerts["date"] == d]["total_score"].values[0]
    clusters.append({
        "start": cluster_start,
        "end": cluster_end,
        "days": (cluster_end - cluster_start).days + 1,
        "max_score": cluster_max_score
    })

# 各クラスタの的中判定
cluster_hits = 0
for c in clusters:
    end_check = c["end"] + timedelta(days=WINDOW)
    occurred = ((m7["date"] >= c["start"]) & (m7["date"] <= end_check)).any()
    c["hit"] = occurred
    if occurred:
        cluster_hits += 1
        # どの地震にヒットしたか
        matches = m7[(m7["date"] >= c["start"]) & (m7["date"] <= end_check)]
        c["matched"] = "; ".join(f"{r['date'].date()} {r['place']} M{r['magnitude']}" for _, r in matches.iterrows())
    else:
        c["matched"] = ""

n_clusters = len(clusters)
cluster_hit_rate = cluster_hits / n_clusters if n_clusters > 0 else 0

print(f"  警報クラスタ数（独立イベント）: {n_clusters}")
print(f"  的中クラスタ数: {cluster_hits}")
print(f"  独立的中率: {cluster_hit_rate:.4f} ({cluster_hit_rate*100:.2f}%)")
print()
for i, c in enumerate(clusters):
    tag = "★的中" if c["hit"] else "  空振り"
    print(f"  クラスタ{i+1}: {c['start'].date()} ～ {c['end'].date()} ({c['days']}日間) "
          f"最大スコア={c['max_score']:.2f} {tag}")
    if c["matched"]:
        print(f"           → {c['matched']}")

# ================================================================
# 2. 二項検定（独立クラスタベース）
# ================================================================
print("\n" + "=" * 70)
print("■ 2. 二項検定（独立クラスタベース）")

# ベースレート: 任意の30日ウィンドウでM7が1回以上発生する確率
m7_count = len(m7)
p_daily = m7_count / total_days
p_window = 1 - (1 - p_daily) ** WINDOW
print(f"  1日あたりM7発生確率: {p_daily:.6f}")
print(f"  30日ウィンドウ内のM7発生確率（ベースレート）: {p_window:.4f} ({p_window*100:.2f}%)")

# 二項検定: n_clusters回中cluster_hits回の成功がp_windowから期待されるより多いか
binom_p = 1 - stats.binom.cdf(cluster_hits - 1, n_clusters, p_window)
print(f"\n  H0: 的中率 = ベースレート ({p_window:.4f})")
print(f"  H1: 的中率 > ベースレート（片側検定）")
print(f"  観測: {n_clusters}回中{cluster_hits}回的中")
print(f"  期待値: {n_clusters * p_window:.2f}回")
print(f"  二項検定 p値: {binom_p:.6f}")
if binom_p < 0.05:
    print(f"  → p < 0.05 で統計的に有意")
else:
    print(f"  → p >= 0.05 で統計的に有意とは言えない")

# ================================================================
# 3. フィッシャーの正確確率検定（2×2分割表）
# ================================================================
print("\n" + "=" * 70)
print("■ 3. フィッシャーの正確確率検定")

# M7イベントの独立化（同日の地震は1イベント）
m7_unique_dates = m7["date"].dt.date.unique()
n_m7_events = len(m7_unique_dates)

# 捕捉されたM7イベント数（独立）
caught_m7 = set()
for c in clusters:
    if c["hit"]:
        end_check = c["end"] + timedelta(days=WINDOW)
        matches = m7[(m7["date"] >= c["start"]) & (m7["date"] <= end_check)]
        for d in matches["date"].dt.date:
            caught_m7.add(d)
n_caught_m7 = len(caught_m7)

# 2×2分割表
#                    M7あり    M7なし
# 警報あり            a         b
# 警報なし            c         d

# 日ベースではなくイベントベースで考える
# ただし、正確な分割表は「日」を単位にする方が妥当
# 警報ウィンドウがカバーする日数
alert_window_days = set()
for c in clusters:
    for i in range((c["end"] - c["start"]).days + WINDOW + 1):
        alert_window_days.add((c["start"] + timedelta(days=i)).date())

m7_date_set = set(m7_unique_dates)
all_dates = set()
for i in range(total_days):
    all_dates.add((score["date"].min() + timedelta(days=i)).date())

a = len(m7_date_set & alert_window_days)  # 警報ウィンドウ内のM7日
b = len(alert_window_days - m7_date_set)  # 警報ウィンドウ内の非M7日
c = len(m7_date_set - alert_window_days)  # 警報なしのM7日
d = len(all_dates - alert_window_days - m7_date_set)  # 警報なし・M7なし

print(f"\n  2×2分割表（日ベース）:")
print(f"                  M7あり    M7なし    合計")
print(f"  警報ウィンドウ内  {a:>6}    {b:>6}    {a+b:>6}")
print(f"  警報ウィンドウ外  {c:>6}    {d:>6}    {c+d:>6}")
print(f"  合計              {a+c:>6}    {b+d:>6}    {a+b+c+d:>6}")

oddsratio, fisher_p = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
print(f"\n  オッズ比: {oddsratio:.4f}")
print(f"  フィッシャーの正確確率検定 p値: {fisher_p:.6f}")
if fisher_p < 0.05:
    print(f"  → p < 0.05 で統計的に有意")
else:
    print(f"  → p >= 0.05 で統計的に有意とは言えない")

# ================================================================
# 4. モンテカルロシミュレーション
# ================================================================
print("\n" + "=" * 70)
print("■ 4. モンテカルロシミュレーション（10万回）")
print("  （ランダムにn_clusters個の警報を出した場合の的中率分布）")

np.random.seed(42)
n_sim = 100_000
sim_hits = np.zeros(n_sim)

all_dates_list = sorted(list(all_dates))
m7_date_set_for_sim = m7_date_set

for i in range(n_sim):
    # ランダムにn_clusters個の日を選んで警報とする
    random_alert_indices = np.random.choice(len(all_dates_list), size=n_clusters, replace=False)
    hits = 0
    for idx in random_alert_indices:
        alert_date = all_dates_list[idx]
        # 30日ウィンドウ内にM7があるか
        for j in range(WINDOW + 1):
            check_date = alert_date + timedelta(days=j)
            if check_date in m7_date_set_for_sim:
                hits += 1
                break
    sim_hits[i] = hits

sim_hit_rates = sim_hits / n_clusters
observed_hit_rate = cluster_hit_rate
p_mc = np.mean(sim_hits >= cluster_hits)

print(f"  観測された的中数: {cluster_hits}/{n_clusters}")
print(f"  シミュレーション平均的中数: {np.mean(sim_hits):.2f}")
print(f"  シミュレーション中で観測値以上の割合: {p_mc:.6f}")
print(f"  → モンテカルロ p値: {p_mc:.6f}")
if p_mc < 0.05:
    print(f"  → p < 0.05 で統計的に有意")
else:
    print(f"  → p >= 0.05 で統計的に有意とは言えない")

# パーセンタイル分布
print(f"\n  シミュレーション的中数分布:")
for k in range(n_clusters + 1):
    pct = np.mean(sim_hits == k) * 100
    if pct > 0.01:
        marker = " ◀ 観測値" if k == cluster_hits else ""
        print(f"    {k}回: {pct:.2f}%{marker}")

# ================================================================
# 5. 効果量（リフト、相対リスク）
# ================================================================
print("\n" + "=" * 70)
print("■ 5. 効果量")

lift = cluster_hit_rate / p_window if p_window > 0 else float('inf')
print(f"  ベースレート: {p_window:.4f} ({p_window*100:.2f}%)")
print(f"  観測的中率（独立クラスタ）: {cluster_hit_rate:.4f} ({cluster_hit_rate*100:.2f}%)")
print(f"  リフト（観測/ベースレート）: {lift:.2f}倍")

# 95%信頼区間（Wilson法）
from statsmodels.stats.proportion import proportion_confint
ci_low, ci_high = proportion_confint(cluster_hits, n_clusters, alpha=0.05, method='wilson')
print(f"  95%信頼区間（Wilson法）: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"                          [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
if ci_low > p_window:
    print(f"  → 信頼区間の下限 > ベースレート → 有意に高い")
else:
    print(f"  → 信頼区間の下限 ≤ ベースレート → 信頼区間がベースレートを含む")

# ================================================================
# 6. 総合判定
# ================================================================
print("\n" + "=" * 70)
print("■ 6. 総合判定")
print("=" * 70)

print(f"""
  検定手法                     p値         有意性
  ──────────────────────────────────────────────
  二項検定（独立クラスタ）     {binom_p:.6f}    {"有意" if binom_p < 0.05 else "非有意"}
  フィッシャーの正確確率検定   {fisher_p:.6f}    {"有意" if fisher_p < 0.05 else "非有意"}
  モンテカルロ（10万回）       {p_mc:.6f}    {"有意" if p_mc < 0.05 else "非有意"}
  ──────────────────────────────────────────────

  独立警報クラスタ数: {n_clusters}（サンプル数が少ない点に注意）
  リフト: {lift:.2f}倍
  95%CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%] vs ベースレート {p_window*100:.2f}%
""")

# 注意事項
print("  【注意事項】")
print("  ・独立クラスタ数が少ない（n={n_clusters}）ため、検定力が低い".format(n_clusters=n_clusters))
print("  ・的中が2024年以降に集中しており、時期依存性がある")
print("  ・同一地域（カムチャツカ）の地震に的中が偏っている")
print("  ・これらはモデルの物理的意味を支持する可能性もあるが、")
print("    過学習やデータスヌーピングの可能性も否定できない")
