import pandas as pd
from datetime import timedelta

# --- 読み込み ---
score = pd.read_csv(r"データ分析結果出力\total_with_gnss.csv", parse_dates=["date"])
eq = pd.read_csv(r"生データ\2010-all.csv", parse_dates=["date"])

# ソート
score = score.sort_values("date").reset_index(drop=True)
eq = eq.sort_values("date").reset_index(drop=True)

# --- M7以上の地震を抽出 ---
m7 = eq[eq["magnitude"] >= 7.0].copy()
print("=== M7以上の地震一覧 ===")
print(m7[["date", "place", "magnitude"]].to_string(index=False))
print(f"\nM7以上の地震回数: {len(m7)}")
print(f"期間: {score['date'].min().date()} ～ {score['date'].max().date()}")

# --- 条件 ---
THRESHOLD = 18
WINDOW = 30  # 日

# スコア18以上の日を抽出
alerts = score[score["total_score"] >= THRESHOLD].copy()

hit_count = 0
results = []

for _, row in alerts.iterrows():
    start = row["date"]
    end = start + timedelta(days=WINDOW)
    # 30日以内にM7があるか
    matches = m7[(m7["date"] >= start) & (m7["date"] <= end)]
    occurred = len(matches) > 0
    if occurred:
        hit_count += 1
    results.append({
        "alert_date": start,
        "score": row["total_score"],
        "m7_within_30days": occurred,
        "matched_earthquakes": "; ".join(
            f"{r['date'].date()} {r['place']} M{r['magnitude']}"
            for _, r in matches.iterrows()
        ) if occurred else ""
    })

results_df = pd.DataFrame(results)

# --- 成功率 ---
total_alerts = len(alerts)
hit_rate = hit_count / total_alerts if total_alerts > 0 else 0

print(f"\n=== 警報モデル評価 (閾値={THRESHOLD}, ウィンドウ={WINDOW}日) ===")
print(f"警報回数: {total_alerts}")
print(f"30日以内M7発生（的中）: {hit_count}")
print(f"的中率: {hit_rate:.4f} ({hit_rate*100:.2f}%)")

# --- 逆方向: M7のうち、事前に警報が出ていたか（捕捉率） ---
caught_count = 0
m7_results = []
for _, row in m7.iterrows():
    eq_date = row["date"]
    # 地震の30日前～当日にスコア18以上の警報があったか
    lookback_start = eq_date - timedelta(days=WINDOW)
    prior_alerts = alerts[(alerts["date"] >= lookback_start) & (alerts["date"] <= eq_date)]
    caught = len(prior_alerts) > 0
    if caught:
        caught_count += 1
    m7_results.append({
        "earthquake_date": eq_date,
        "place": row["place"],
        "magnitude": row["magnitude"],
        "alert_within_prior_30days": caught,
        "prior_alert_count": len(prior_alerts),
        "prior_alert_dates": "; ".join(str(d.date()) for d in prior_alerts["date"]) if caught else ""
    })

m7_results_df = pd.DataFrame(m7_results)
total_m7 = len(m7)
catch_rate = caught_count / total_m7 if total_m7 > 0 else 0

print(f"\n=== 捕捉率（M7側から見た評価） ===")
print(f"M7地震回数: {total_m7}")
print(f"事前警報あり: {caught_count}")
print(f"捕捉率: {catch_rate:.4f} ({catch_rate*100:.2f}%)")

# --- ベースレート比較 ---
total_days = (score["date"].max() - score["date"].min()).days + 1
alert_days = total_alerts
alert_ratio = alert_days / total_days if total_days > 0 else 0
print(f"\n=== ベースレート ===")
print(f"総日数: {total_days}")
print(f"警報日数: {alert_days}")
print(f"警報発令率: {alert_ratio:.4f} ({alert_ratio*100:.2f}%)")

# ランダムに同じ割合で警報を出した場合の期待的中率
# 30日ウィンドウで少なくとも1つのM7をキャッチする確率
m7_days = len(m7["date"].dt.date.unique())
# 各警報日の30日ウィンドウがカバーする日数を考慮
# 簡易計算: M7が存在する30日ウィンドウの割合
print(f"M7発生日数（ユニーク）: {m7_days}")

# 保存
results_df.to_csv(r"データ分析結果出力\verification_alert_result.csv", index=False)
m7_results_df.to_csv(r"データ分析結果出力\verification_m7_catch.csv", index=False)
print(f"\n結果保存: verification_alert_result.csv, verification_m7_catch.csv")

# --- 詳細表示 ---
print("\n=== 警報詳細（的中のみ） ===")
hits = results_df[results_df["m7_within_30days"] == True]
if len(hits) > 0:
    print(hits.to_string(index=False))
else:
    print("的中なし")

print("\n=== M7捕捉詳細 ===")
print(m7_results_df.to_string(index=False))
