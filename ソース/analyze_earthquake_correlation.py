"""
地震スコアと大地震の相関分析スクリプト

M7以上の日本国内地震に対して、スコアが前兆として機能しているかを分析する
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "生データ"
OUTPUT_DIR = PROJECT_ROOT / "データ分析結果出力"


def load_earthquake_data():
    """国内地震データを読み込み"""
    japan_file = RAW_DATA_DIR / "2010-all.csv"
    df = pd.read_csv(japan_file)

    # 日付変換
    def parse_date(date_str):
        if '月' in str(date_str) and '日' in str(date_str) and '年' not in str(date_str):
            return pd.to_datetime(f'2010年{date_str}', format='%Y年%m月%d日')
        else:
            return pd.to_datetime(date_str, format='%Y/%m/%d')

    df['date'] = df['date'].apply(parse_date)
    df['magnitude'] = pd.to_numeric(df['magnitude'], errors='coerce')
    return df


def load_score_data(filename):
    """スコアデータを読み込み"""
    score_file = OUTPUT_DIR / filename
    if not score_file.exists():
        return None
    df = pd.read_csv(score_file, parse_dates=['date'])
    return df


def analyze_precursor(score_df, eq_date, days_before=30):
    """
    地震発生前のスコアを分析

    Parameters:
    -----------
    score_df : DataFrame
        スコアデータ
    eq_date : datetime
        地震発生日
    days_before : int
        何日前までを分析対象とするか

    Returns:
    --------
    dict : 分析結果
    """
    # 地震発生前30日間のスコア
    start_date = eq_date - timedelta(days=days_before)
    end_date = eq_date - timedelta(days=1)  # 当日は除く

    mask = (score_df['date'] >= start_date) & (score_df['date'] <= end_date)
    precursor_scores = score_df[mask]['total_score']

    if len(precursor_scores) == 0:
        return None

    # 全期間の統計
    all_scores = score_df['total_score']
    all_mean = all_scores.mean()
    all_std = all_scores.std()

    # 前兆期間の統計
    precursor_mean = precursor_scores.mean()
    precursor_max = precursor_scores.max()
    precursor_min = precursor_scores.min()

    # 高スコア日数（平均+1σ以上）
    threshold_1sigma = all_mean + all_std
    high_score_days = (precursor_scores >= threshold_1sigma).sum()

    # 高スコア日数（平均+2σ以上）
    threshold_2sigma = all_mean + 2 * all_std
    very_high_score_days = (precursor_scores >= threshold_2sigma).sum()

    return {
        'precursor_mean': precursor_mean,
        'precursor_max': precursor_max,
        'precursor_min': precursor_min,
        'high_score_days_1sigma': high_score_days,
        'very_high_score_days_2sigma': very_high_score_days,
        'all_mean': all_mean,
        'all_std': all_std,
        'threshold_1sigma': threshold_1sigma,
        'threshold_2sigma': threshold_2sigma,
        'z_score': (precursor_mean - all_mean) / all_std if all_std > 0 else 0
    }


def main():
    print("=" * 70)
    print("地震スコアと大地震の相関分析")
    print("=" * 70)

    # 1. データ読み込み
    print("\n[1] データ読み込み...")
    eq_df = load_earthquake_data()

    # M7以上の地震を抽出
    major_eq = eq_df[eq_df['magnitude'] >= 7.0].copy()
    major_eq = major_eq.sort_values('date')

    print(f"\nM7以上の日本国内地震: {len(major_eq)}件")
    print("-" * 70)
    print(f"{'日付':<12} {'場所':<20} {'M':>4}")
    print("-" * 70)
    for _, row in major_eq.iterrows():
        print(f"{row['date'].strftime('%Y-%m-%d'):<12} {row['place']:<20} {row['magnitude']:>4.1f}")

    # 2. スコアデータ読み込み
    print("\n[2] スコアデータ読み込み...")

    # GNSSあり (total.csv = 旧形式、または total_with_gnss.csv)
    score_with_gnss = load_score_data("total.csv")
    if score_with_gnss is None:
        score_with_gnss = load_score_data("total_with_gnss.csv")

    # GNSSなし
    score_no_gnss = load_score_data("total_no_gnss.csv")

    # 3. 各大地震に対する前兆分析
    print("\n[3] 前兆スコア分析（地震発生前30日間）")

    results = []

    for _, eq in major_eq.iterrows():
        eq_date = eq['date']
        eq_place = eq['place']
        eq_mag = eq['magnitude']

        result = {
            'date': eq_date,
            'place': eq_place,
            'magnitude': eq_mag
        }

        # GNSSありの分析
        if score_with_gnss is not None:
            analysis = analyze_precursor(score_with_gnss, eq_date)
            if analysis:
                result['gnss_precursor_mean'] = analysis['precursor_mean']
                result['gnss_precursor_max'] = analysis['precursor_max']
                result['gnss_high_days_1sigma'] = analysis['high_score_days_1sigma']
                result['gnss_z_score'] = analysis['z_score']

        # GNSSなしの分析
        if score_no_gnss is not None:
            analysis = analyze_precursor(score_no_gnss, eq_date)
            if analysis:
                result['no_gnss_precursor_mean'] = analysis['precursor_mean']
                result['no_gnss_precursor_max'] = analysis['precursor_max']
                result['no_gnss_high_days_1sigma'] = analysis['high_score_days_1sigma']
                result['no_gnss_z_score'] = analysis['z_score']

        results.append(result)

    # 結果をDataFrameに
    results_df = pd.DataFrame(results)

    # 4. 結果表示
    print("\n" + "=" * 70)
    print("【分析結果】M7以上地震の前兆スコア比較")
    print("=" * 70)
    print(f"{'日付':<12} {'場所':<16} {'M':>4} | {'GNSSあり':^20} | {'GNSSなし':^20}")
    print(f"{'':12} {'':16} {'':4} | {'平均':>6} {'最大':>6} {'Z値':>6} | {'平均':>6} {'最大':>6} {'Z値':>6}")
    print("-" * 70)

    for _, r in results_df.iterrows():
        gnss_mean = r.get('gnss_precursor_mean', 0)
        gnss_max = r.get('gnss_precursor_max', 0)
        gnss_z = r.get('gnss_z_score', 0)
        no_gnss_mean = r.get('no_gnss_precursor_mean', 0)
        no_gnss_max = r.get('no_gnss_precursor_max', 0)
        no_gnss_z = r.get('no_gnss_z_score', 0)

        place_short = r['place'][:14] if len(r['place']) > 14 else r['place']
        print(f"{r['date'].strftime('%Y-%m-%d'):<12} {place_short:<16} {r['magnitude']:>4.1f} | "
              f"{gnss_mean:>6.2f} {gnss_max:>6.2f} {gnss_z:>6.2f} | "
              f"{no_gnss_mean:>6.2f} {no_gnss_max:>6.2f} {no_gnss_z:>6.2f}")

    # 5. 統計サマリー
    print("\n" + "=" * 70)
    print("【統計サマリー】")
    print("=" * 70)

    if score_with_gnss is not None:
        all_mean_gnss = score_with_gnss['total_score'].mean()
        all_std_gnss = score_with_gnss['total_score'].std()
        print(f"\nGNSSあり全期間: 平均={all_mean_gnss:.2f}, 標準偏差={all_std_gnss:.2f}")
        print(f"  1σ閾値: {all_mean_gnss + all_std_gnss:.2f}")
        print(f"  2σ閾値: {all_mean_gnss + 2*all_std_gnss:.2f}")

        if 'gnss_z_score' in results_df.columns:
            avg_z = results_df['gnss_z_score'].mean()
            positive_z_count = (results_df['gnss_z_score'] > 0).sum()
            print(f"  M7前兆期間の平均Z値: {avg_z:.2f}")
            print(f"  Z値>0の地震数: {positive_z_count}/{len(results_df)} ({positive_z_count/len(results_df)*100:.1f}%)")

    if score_no_gnss is not None:
        all_mean_no_gnss = score_no_gnss['total_score'].mean()
        all_std_no_gnss = score_no_gnss['total_score'].std()
        print(f"\nGNSSなし全期間: 平均={all_mean_no_gnss:.2f}, 標準偏差={all_std_no_gnss:.2f}")
        print(f"  1σ閾値: {all_mean_no_gnss + all_std_no_gnss:.2f}")
        print(f"  2σ閾値: {all_mean_no_gnss + 2*all_std_no_gnss:.2f}")

        if 'no_gnss_z_score' in results_df.columns:
            avg_z = results_df['no_gnss_z_score'].mean()
            positive_z_count = (results_df['no_gnss_z_score'] > 0).sum()
            print(f"  M7前兆期間の平均Z値: {avg_z:.2f}")
            print(f"  Z値>0の地震数: {positive_z_count}/{len(results_df)} ({positive_z_count/len(results_df)*100:.1f}%)")

    # 6. 検出率分析
    print("\n" + "=" * 70)
    print("【検出率分析】前兆期間にスコアが閾値を超えた割合")
    print("=" * 70)

    for threshold_name, threshold_sigma in [("1σ超え", 1), ("2σ超え", 2)]:
        print(f"\n{threshold_name}の検出率:")

        if score_with_gnss is not None:
            threshold = score_with_gnss['total_score'].mean() + threshold_sigma * score_with_gnss['total_score'].std()
            detected = 0
            for _, eq in major_eq.iterrows():
                eq_date = eq['date']
                start_date = eq_date - timedelta(days=30)
                end_date = eq_date - timedelta(days=1)
                mask = (score_with_gnss['date'] >= start_date) & (score_with_gnss['date'] <= end_date)
                if (score_with_gnss[mask]['total_score'] >= threshold).any():
                    detected += 1
            print(f"  GNSSあり: {detected}/{len(major_eq)} ({detected/len(major_eq)*100:.1f}%)")

        if score_no_gnss is not None:
            threshold = score_no_gnss['total_score'].mean() + threshold_sigma * score_no_gnss['total_score'].std()
            detected = 0
            for _, eq in major_eq.iterrows():
                eq_date = eq['date']
                start_date = eq_date - timedelta(days=30)
                end_date = eq_date - timedelta(days=1)
                mask = (score_no_gnss['date'] >= start_date) & (score_no_gnss['date'] <= end_date)
                if (score_no_gnss[mask]['total_score'] >= threshold).any():
                    detected += 1
            print(f"  GNSSなし: {detected}/{len(major_eq)} ({detected/len(major_eq)*100:.1f}%)")

    # 7. チューニング提案
    print("\n" + "=" * 70)
    print("【チューニング提案】")
    print("=" * 70)
    print("""
1. 前震パターンの強化
   - M5-M6.9の地震が3日以内に複数回発生した場合のスコア加算
   - 同一地域での地震集中（クラスタリング）の検出

2. 時間窓の最適化
   - 現在30日窓 → 7日、14日、30日の複数窓で評価
   - 短期的な急変（3日以内の急上昇）を重視

3. 地域重み付け
   - 過去のM7以上発生地域（南海トラフ、相模トラフ等）の重み増加
   - 深発地震（300km以上）が特定地域で発生した場合の加算

4. 静穏期検出の改善
   - 「地震が少ない」期間の検出精度向上
   - 静穏期後の急増パターンをより重視

5. b値（グーテンベルク・リヒター則）の導入
   - b値の低下は大地震の前兆とされる
   - 日次b値を計算してスコアに反映
""")

    # 結果をCSV保存
    output_file = OUTPUT_DIR / "earthquake_correlation_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n分析結果を保存しました: {output_file}")


if __name__ == "__main__":
    main()
