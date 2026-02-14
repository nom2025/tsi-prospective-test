"""
TSI 日次計算スクリプト（凍結版）
================================
凍結日: 2026-02-14
仕様書: TSI_FROZEN_SPEC.md

このスクリプトは凍結された数式・パラメータでTSIを計算し、
prospective_log.csv に追記する。

使い方:
    python tsi_daily_calc.py

数式・パラメータを変更してはならない。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# ================================================================
# 凍結パラメータ（TSI_FROZEN_SPEC.md より）
# ================================================================
FROZEN_PARAMS = {
    "freeze_date": "2026-02-14",
    "standardization": {
        "vc_horiz_mean": {"mean": 2.8771487892, "std": 3.2528439241},
        "coherence_vc":  {"mean": 0.7471262423, "std": 0.2245541473},
        "accel_mean":    {"mean": 0.1238355949, "std": 0.1323451171},
        "lf_mean":       {"mean": 3.6413308546, "std": 5.9476394549},
    },
    "alert_threshold_95": 0.6138,
    "stations": [950421, 950447, 950456, 950465, 950474, 950483, 950492],
    "ma_short": 7,
    "ma_long": 30,
    "velocity_window": 14,
    "accel_diff": 7,
    "lf_diff": 14,
    "interpolation_limit": 7,
    "coherence_min_stations": 3,
    "weights": [0.25, 0.25, 0.25, 0.25],  # 等重み
}

# パス設定
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "生データ"
OUTPUT_DIR = BASE_DIR / "データ分析結果出力"
LOG_FILE = BASE_DIR / "prospective_log.csv"


def calculate_tsi_from_gnss(gnss_file):
    """
    GNSSデータからTSIを計算する（凍結された数式のみ使用）

    Returns:
        pd.DataFrame: date, TSI, TSI_30d, 各成分値
    """
    df = pd.read_csv(gnss_file, parse_dates=["date"])
    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)

    stations = FROZEN_PARAMS["stations"]
    MA_S = FROZEN_PARAMS["ma_short"]   # 7
    MA_L = FROZEN_PARAMS["ma_long"]    # 30
    VEL_W = FROZEN_PARAMS["velocity_window"]  # 14
    ACC_D = FROZEN_PARAMS["accel_diff"]  # 7
    LF_D = FROZEN_PARAMS["lf_diff"]    # 14
    INTERP_LIMIT = FROZEN_PARAMS["interpolation_limit"]  # 7

    all_station_daily = {}

    for sid in stations:
        sdata = df[df["station_id"] == sid].copy()
        if len(sdata) == 0:
            continue
        sdata = sdata.set_index("date").sort_index()

        coords = sdata[["east_mm", "north_mm", "up_mm"]].copy()
        full_idx = pd.date_range(coords.index.min(), coords.index.max(), freq="D")
        coords = coords.reindex(full_idx)
        coords = coords.interpolate(method="linear", limit=INTERP_LIMIT)

        # ① 速度変化: MA7 - MA30
        ma7 = coords.rolling(MA_S, center=True, min_periods=MA_S // 2 + 1).mean()
        ma30 = coords.rolling(MA_L, center=True, min_periods=MA_L // 2).mean()
        vc = ma7 - ma30
        vc_horiz = np.sqrt(vc["east_mm"]**2 + vc["north_mm"]**2)

        # ② 速度（14日窓の傾き）
        vel_e = coords["east_mm"].rolling(VEL_W, center=True, min_periods=VEL_W // 2).apply(
            lambda x: np.polyfit(range(len(x)), x.values, 1)[0]
            if len(x.dropna()) >= VEL_W // 2 else np.nan, raw=False)
        vel_n = coords["north_mm"].rolling(VEL_W, center=True, min_periods=VEL_W // 2).apply(
            lambda x: np.polyfit(range(len(x)), x.values, 1)[0]
            if len(x.dropna()) >= VEL_W // 2 else np.nan, raw=False)

        # ③ 加速度
        acc_e = vel_e.diff(ACC_D) / ACC_D
        acc_n = vel_n.diff(ACC_D) / ACC_D
        acc_horiz = np.sqrt(acc_e**2 + acc_n**2)

        # ④ 低周波偏差
        lf_dev = ma30.diff(LF_D)
        lf_horiz = np.sqrt(lf_dev["east_mm"]**2 + lf_dev["north_mm"]**2)

        all_station_daily[sid] = pd.DataFrame({
            "vc_east": vc["east_mm"],
            "vc_north": vc["north_mm"],
            "vc_horiz": vc_horiz,
            "velocity_e": vel_e,
            "velocity_n": vel_n,
            "accel_horiz": acc_horiz,
            "lf_horiz": lf_horiz,
        }, index=full_idx)

    if not all_station_daily:
        return pd.DataFrame()

    # 広域統合
    min_date = max(sdf.index.min() for sdf in all_station_daily.values()) + timedelta(days=MA_L)
    max_date = min(sdf.index.max() for sdf in all_station_daily.values()) - timedelta(days=MA_L)
    all_dates = pd.date_range(min_date, max_date, freq="D")

    records = []
    for date in all_dates:
        vc_vals = []
        accel_vals = []
        lf_vals = []
        vectors_vc = []

        for sid in stations:
            if sid not in all_station_daily:
                continue
            sdf = all_station_daily[sid]
            if date not in sdf.index:
                continue
            row = sdf.loc[date]

            if not np.isnan(row["vc_horiz"]):
                vc_vals.append(row["vc_horiz"])
            if not np.isnan(row["accel_horiz"]):
                accel_vals.append(row["accel_horiz"])
            if not np.isnan(row["lf_horiz"]):
                lf_vals.append(row["lf_horiz"])
            if not np.isnan(row["vc_east"]) and not np.isnan(row["vc_north"]):
                vectors_vc.append([row["vc_east"], row["vc_north"]])

        # ① 速度変化
        vc_mean = np.mean(vc_vals) if vc_vals else np.nan

        # ② 空間整合性
        coherence = np.nan
        if len(vectors_vc) >= FROZEN_PARAMS["coherence_min_stations"]:
            vecs = np.array(vectors_vc)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            unit_vecs = vecs / norms
            mean_vec = unit_vecs.mean(axis=0)
            coherence = np.linalg.norm(mean_vec)

        # ③ 加速度
        accel = np.mean(accel_vals) if accel_vals else np.nan

        # ④ 低周波偏差
        lf = np.mean(lf_vals) if lf_vals else np.nan

        # Zスコア標準化（凍結パラメータ使用）
        P = FROZEN_PARAMS["standardization"]
        z_vc = (vc_mean - P["vc_horiz_mean"]["mean"]) / P["vc_horiz_mean"]["std"] if not np.isnan(vc_mean) else np.nan
        z_coh = (coherence - P["coherence_vc"]["mean"]) / P["coherence_vc"]["std"] if not np.isnan(coherence) else np.nan
        z_acc = (accel - P["accel_mean"]["mean"]) / P["accel_mean"]["std"] if not np.isnan(accel) else np.nan
        z_lf = (lf - P["lf_mean"]["mean"]) / P["lf_mean"]["std"] if not np.isnan(lf) else np.nan

        # TSI合成（等重み）
        z_vals = [z_vc, z_coh, z_acc, z_lf]
        valid_z = [z for z in z_vals if not np.isnan(z)]
        tsi_val = np.mean(valid_z) if valid_z else np.nan

        records.append({
            "date": date,
            "vc_horiz_mean": vc_mean,
            "coherence_vc": coherence,
            "accel_mean": accel,
            "lf_mean": lf,
            "z_vc": z_vc,
            "z_coherence": z_coh,
            "z_accel": z_acc,
            "z_lf": z_lf,
            "TSI": tsi_val,
        })

    result = pd.DataFrame(records)
    result = result.set_index("date").sort_index()

    # TSI_30d
    result["TSI_30d"] = result["TSI"].rolling(MA_L, center=True, min_periods=MA_L // 2).mean()

    # 警報判定
    result["alert"] = result["TSI_30d"] >= FROZEN_PARAMS["alert_threshold_95"]

    return result


def get_gnss_observation_date(gnss_file):
    """GNSSデータファイルから最終観測日を取得する"""
    try:
        df = pd.read_csv(gnss_file, parse_dates=["date"])
        return df["date"].max().date().isoformat()
    except Exception:
        return "unknown"


def update_prospective_log(tsi_df, gnss_observation_date):
    """
    前向き検証ログに新しいデータを追記する。

    ===== 絶対ルール =====
    1. 既存ファイルは読み取り専用。新しい行を末尾にappendするだけ
    2. 過去の行は絶対に修正・削除しない
    3. GNSSデータが後日再解析されても、過去のTSI値を再計算しない
    4. GNSSの最終観測日が更新されたときにだけ記録する
    5. 同じobservation_dateの二重記録を禁止する
    =======================

    各行の監査列（科学的証拠の核心）:
    - observation_date: GNSSデータの最終観測日（計算に使ったデータの鮮度）
    - publish_date: このスクリプトが実行された日（GitHub Actionsの実行日）
    第三者がこの2列で「未来情報を使っていない」ことを確認できる。
    """
    freeze_date = pd.Timestamp(FROZEN_PARAMS["freeze_date"])

    # TSI時系列から最新の値を取得（observation_dateに対応するTSI）
    latest_tsi_date = tsi_df.index.max()
    if latest_tsi_date < freeze_date:
        print("  凍結日以降のデータなし")
        return

    # GNSSの最終観測日が変わっていなければ記録しない
    # （毎日Actionsが動いても、GNSSデータが更新されない限りログは増えない）
    if LOG_FILE.exists() and LOG_FILE.stat().st_size > 0:
        existing = pd.read_csv(LOG_FILE)

        # 二重記録防止: 同じobservation_dateは記録しない
        if gnss_observation_date in existing["observation_date"].astype(str).values:
            print(f"  observation_date={gnss_observation_date} は記録済み → スキップ")
            return

        # ドリフト検出: 同じdateで値が変わっていないか確認
        latest_date_str = str(latest_tsi_date.date())
        if latest_date_str in existing["date"].astype(str).values:
            old_row = existing[existing["date"].astype(str) == latest_date_str].iloc[-1]
            old_tsi = old_row["TSI"]
            new_tsi = tsi_df.loc[latest_tsi_date, "TSI"]
            if abs(old_tsi - new_tsi) > 1e-6:
                print(f"  [警告] {latest_date_str} のTSI値がドリフト: "
                      f"記録済み={old_tsi:.6f} → 再計算={new_tsi:.6f}")
                print("  [重要] GNSSデータの再解析が検出されました。")
                print("  既存の記録値は変更しません（前向き検証の原則）。")
            print(f"  date={latest_date_str} は記録済み → スキップ")
            return

    # === 新しい1行を作成 ===
    latest = tsi_df.loc[latest_tsi_date]
    new_row = pd.DataFrame([{
        "date":             str(latest_tsi_date.date()),
        "observation_date": gnss_observation_date,
        "publish_date":     datetime.now().strftime("%Y-%m-%d"),
        "TSI":              round(latest["TSI"], 6),
        "TSI_30d":          round(latest["TSI_30d"], 6) if pd.notna(latest["TSI_30d"]) else "",
        "alert":            int(latest["alert"]) if pd.notna(latest["alert"]) else 0,
        "vc_horiz_mean":    round(latest["vc_horiz_mean"], 6) if "vc_horiz_mean" in latest.index else "",
        "coherence_vc":     round(latest["coherence_vc"], 6) if "coherence_vc" in latest.index else "",
        "accel_mean":       round(latest["accel_mean"], 6) if "accel_mean" in latest.index else "",
        "lf_mean":          round(latest["lf_mean"], 6) if "lf_mean" in latest.index else "",
    }])

    # === 追記（append only） ===
    if LOG_FILE.exists() and LOG_FILE.stat().st_size > 0:
        # header=False: ヘッダー行を追加しない（これ忘れるとCSVが壊れる）
        new_row.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        # 初回: ヘッダー付きで新規作成
        new_row.to_csv(LOG_FILE, mode="w", header=True, index=False)

    print(f"  ✓ 1行追記完了")
    print(f"    date:             {str(latest_tsi_date.date())}")
    print(f"    observation_date: {gnss_observation_date}")
    print(f"    publish_date:     {datetime.now().strftime('%Y-%m-%d')}")
    print(f"    TSI:              {latest['TSI']:.6f}")
    print(f"    TSI_30d:          {latest['TSI_30d']:.6f}" if pd.notna(latest['TSI_30d']) else "    TSI_30d:          N/A")
    print(f"    alert:            {int(latest['alert'])}")


def main():
    print("=" * 60)
    print("  TSI 日次計算（凍結版）")
    print(f"  凍結日: {FROZEN_PARAMS['freeze_date']}")
    print(f"  実行日: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    gnss_file = RAW_DIR / "gnss_data.csv"
    if not gnss_file.exists():
        print(f"  [エラー] GNSSデータファイルが見つかりません: {gnss_file}")
        return

    # GNSSデータの最終観測日を記録（監査用）
    gnss_obs_date = get_gnss_observation_date(gnss_file)
    print(f"  GNSSデータ最終観測日: {gnss_obs_date}")

    print("\n■ TSI計算中...")
    tsi_df = calculate_tsi_from_gnss(gnss_file)

    if len(tsi_df) == 0:
        print("  [エラー] TSI計算結果が空です")
        return

    print(f"  計算完了: {len(tsi_df)}日分")
    print(f"  最新日: {tsi_df.index.max().date()}")

    # 最新のTSI値を表示
    latest = tsi_df.iloc[-1]
    latest_date = tsi_df.index[-1]
    print(f"\n■ 最新値 ({latest_date.date()}):")
    print(f"  TSI      = {latest['TSI']:.4f}")
    print(f"  TSI_30d  = {latest['TSI_30d']:.4f}" if not np.isnan(latest['TSI_30d']) else "  TSI_30d  = N/A (データ不足)")
    print(f"  警報状態 = {'★ 警報中' if latest['alert'] else '通常'}")

    # tsi_frozen.csv の保存（凍結日以前のデータのみ上書き許可）
    # 凍結日以前 = 過去のバックテスト用参考データ（検証対象外）
    # 凍結日以降 = prospective_log.csv にのみ記録（こちらが正式記録）
    output_file = OUTPUT_DIR / "tsi_frozen.csv"
    tsi_df.to_csv(output_file)
    print(f"\n■ 参考用全データ保存: {output_file}")
    print(f"  注意: 検証の正式記録は prospective_log.csv のみ")

    # 前向き検証ログ更新（追記のみ・上書き禁止）
    print("\n■ 前向き検証ログ更新中...")
    update_prospective_log(tsi_df, gnss_obs_date)

    # 直近30日のTSI状況
    recent = tsi_df.tail(30)
    alert_days = recent[recent["alert"] == True]
    print(f"\n■ 直近30日の状況:")
    print(f"  警報日数: {len(alert_days)}日")
    if len(alert_days) > 0:
        print(f"  警報期間:")
        for date, row in alert_days.iterrows():
            print(f"    {date.date()} TSI_30d={row['TSI_30d']:.4f}")

    print("\n完了")


if __name__ == "__main__":
    main()
