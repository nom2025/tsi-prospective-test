"""
地震データ分析システム - スコア計算スクリプト

このスクリプトは以下の処理を自動実行します：
1. 生データフォルダからデータを読み込み
2. 4種類のスコア（国内地震、世界地震、深さ、GNSS）を計算
3. 総合スコアを計算（M7.5以上の前兆検出を強化）
4. データ分析結果出力フォルダにCSVファイルを出力

実行方法:
    python ソース/calculate_scores.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def calculate_b_value(magnitudes, mc=2.0):
    """
    グーテンベルク・リヒター則に基づくb値を計算する（Aki法）

    b値は地震のマグニチュード分布の傾きを表す指標。
    通常時はb≈1.0だが、大地震前にはb値が低下する傾向がある。

    計算式: b = log10(e) / (mean_M - (Mc - delta_M/2))
    delta_M = 0.1（マグニチュードの離散間隔）

    Parameters:
    -----------
    magnitudes : array-like
        マグニチュードのリスト
    mc : float
        完全性限界（Magnitude of Completeness）。この値以上の地震のみを使用。
        日本の気象庁データでは通常2.0程度

    Returns:
    --------
    float or None
        b値。データが不足している場合はNone
    """
    # Mc以上の地震のみをフィルタリング
    filtered = [m for m in magnitudes if m >= mc]

    # 最低10個のデータが必要（統計的信頼性のため）
    if len(filtered) < 10:
        return None

    # Aki-Utsu法によるb値計算（離散補正あり）
    # b = log10(e) / (mean_M - (Mc - delta_M/2))
    # delta_M = 0.1（気象庁データの離散間隔）
    delta_m = 0.1
    mean_m = np.mean(filtered)
    mc_corrected = mc - delta_m / 2

    # 分母が0または負にならないようにチェック
    if mean_m <= mc_corrected:
        return None

    b = np.log10(np.e) / (mean_m - mc_corrected)

    return b


def calculate_b_value_score(df, current_date, window_days=30, mc=2.0,
                            baseline_b=None, baseline_std=None):
    """
    b値に基づくスコアを計算する

    b値が低下している場合、大地震の前兆の可能性があるためスコアを加算。
    絶対値ではなく、ベースラインからの相対的な低下を評価する。

    Parameters:
    -----------
    df : pd.DataFrame
        地震データ（date, magnitudeカラムが必要）
    current_date : pd.Timestamp
        スコア計算の基準日
    window_days : int
        b値計算に使用する期間（日数）
    mc : float
        完全性限界
    baseline_b : float or None
        ベースラインのb値（全期間の平均）
    baseline_std : float or None
        ベースラインのb値の標準偏差

    Returns:
    --------
    tuple (float, float or None)
        (スコア, b値)
    """
    # 指定期間のデータを取得
    start_date = current_date - pd.Timedelta(days=window_days)
    period_data = df[(df['date'] >= start_date) & (df['date'] <= current_date)]

    if len(period_data) == 0:
        return 0.0, None

    magnitudes = period_data['magnitude'].dropna().tolist()
    b_value = calculate_b_value(magnitudes, mc)

    if b_value is None:
        return 0.0, None

    # b値に基づくスコア計算
    # ベースラインからの相対的な低下を評価
    score = 0.0

    if baseline_b is not None and baseline_std is not None and baseline_std > 0:
        # Z値で評価（ベースラインからの乖離度）
        z_score = (baseline_b - b_value) / baseline_std

        # b値がベースラインより低いほど高スコア
        if z_score >= 2.0:
            score = 2.0  # 2σ以上低い → 高リスク
        elif z_score >= 1.5:
            score = 1.5  # 1.5σ以上低い → 中〜高リスク
        elif z_score >= 1.0:
            score = 1.0  # 1σ以上低い → 中リスク
        elif z_score >= 0.5:
            score = 0.5  # 0.5σ以上低い → 低リスク
    else:
        # ベースラインがない場合は絶対値で評価（フォールバック）
        if b_value < 0.25:
            score = 2.0
        elif b_value < 0.28:
            score = 1.5
        elif b_value < 0.30:
            score = 1.0
        elif b_value < 0.32:
            score = 0.5

    return score, b_value


def calculate_baseline_b_value(df, mc=2.0, window_days=30):
    """
    全期間のb値の統計（ベースライン）を計算する

    Parameters:
    -----------
    df : pd.DataFrame
        地震データ
    mc : float
        完全性限界
    window_days : int
        各時点でのb値計算ウィンドウ

    Returns:
    --------
    tuple (float, float)
        (平均b値, 標準偏差)
    """
    dates = sorted(df['date'].unique())
    b_values = []

    for current_date in dates:
        start_date = current_date - pd.Timedelta(days=window_days)
        period_data = df[(df['date'] >= start_date) & (df['date'] <= current_date)]

        if len(period_data) == 0:
            continue

        magnitudes = period_data['magnitude'].dropna().tolist()
        b_val = calculate_b_value(magnitudes, mc)

        if b_val is not None:
            b_values.append(b_val)

    if len(b_values) == 0:
        return None, None

    return np.mean(b_values), np.std(b_values)


# プロジェクトルートのパスを取得（このスクリプトの位置から相対的に計算）
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "生データ"
OUTPUT_DIR = PROJECT_ROOT / "データ分析結果出力"

# 出力ディレクトリが存在しない場合は作成
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_japan_earthquake_date(date_str):
    """
    国内地震データの日付形式を統一する
    
    Parameters:
    -----------
    date_str : str
        日付文字列（'12月9日' または '2010/1/2' 形式）
    
    Returns:
    --------
    pandas.Timestamp
        変換された日付
    """
    # '月'と'日'が含まれ、'年'が含まれない場合 (例: '12月9日')
    if '月' in str(date_str) and '日' in str(date_str) and '年' not in str(date_str):
        # 2010年からのデータと仮定して年を付与
        return pd.to_datetime(f'2010年{date_str}', format='%Y年%m月%d日')
    else:
        # 通常の'YYYY/MM/DD'形式 (例: '2010/1/2')
        return pd.to_datetime(date_str, format='%Y/%m/%d')


def calculate_daily_score(df):
    """
    日毎に地震データに基づいてスコアを計算する関数（M7.5前兆強化版）
    
    Parameters:
    -----------
    df : pd.DataFrame
        国内地震データ（date, place, magnitudeカラムが必要）
    
    Returns:
    --------
    pd.DataFrame
        日毎のスコア（date, scoreカラム）
    """
    base_date = pd.to_datetime('2010-02-01')
    
    # 日毎のスコアを格納するDataFrame
    daily_scores = pd.DataFrame(index=df['date'].unique(), columns=['score'])
    daily_scores['score'] = 0.0
    daily_scores.index.name = 'date'
    
    # M7.5発生日（M7.5以上の前兆検出を強化）
    m75_dates = df[df['magnitude'] >= 7.5]['date'].unique()
    m7_dates = df[df['magnitude'] >= 7]['date'].unique()
    
    for day in sorted(daily_scores.index):
        daily_data = df[df['date'] == day]
        
        # 条件1: 1日あたりのmagnitude数が3回以内が3日以上連続
        if daily_data['magnitude'].count() <= 3:
            if all(
                ((day - pd.Timedelta(days=i)) in df['date'].values) and
                (df[df['date'] == (day - pd.Timedelta(days=i))]['magnitude'].count() <= 3)
                for i in range(3)
            ):
                daily_scores.loc[day, 'score'] += 1
        
        # 条件2: 同じく6日以上連続
        if daily_data['magnitude'].count() <= 3:
            consecutive_days = 0
            for i in range(6):
                target_day = day - pd.Timedelta(days=i)
                if target_day in df['date'].values and df[df['date'] == target_day]['magnitude'].count() <= 3:
                    consecutive_days += 1
                else:
                    break
            if consecutive_days >= 6:
                daily_scores.loc[day, 'score'] += 2
        
        # 条件3: 当日10回以上 & 過去30日全て6.5以下
        if daily_data['magnitude'].count() >= 10:
            prev_30 = df[(df['date'] >= day - pd.Timedelta(days=30)) & (df['date'] < day)]['magnitude']
            if len(prev_30) > 0 and all(prev_30 <= 6.5):
                daily_scores.loc[day, 'score'] += 1
        
        # 条件4: M5以上が5回以上
        if daily_data[daily_data['magnitude'] >= 5].shape[0] >= 5:
            daily_scores.loc[day, 'score'] += 0.5
        
        # 条件5: M5以上が10回以上
        if daily_data[daily_data['magnitude'] >= 5].shape[0] >= 10:
            daily_scores.loc[day, 'score'] += 1
        
        # 条件6: 特定地域
        target_places = ['日向灘', '択捉島南東沖', '薩摩半島西方沖', '小笠原諸島西方沖', '千島列島', '鳥島近海', '父島近海']
        if daily_data['place'].str.contains('|'.join(target_places), na=False).any():
            daily_scores.loc[day, 'score'] += 1
        
        # 急増スコア
        prev3_avg = df[(df['date'] < day) & (df['date'] >= day - pd.Timedelta(days=3))]['magnitude'].count() / 3
        if prev3_avg >= 1 and daily_data['magnitude'].count() >= prev3_avg * 2:
            daily_scores.loc[day, 'score'] += 1.0
        
        # 局所集中スコア
        place_counts = daily_data['place'].value_counts()
        if (place_counts >= 5).any():
            daily_scores.loc[day, 'score'] += 1.0
        
        # M7.5前兆強化: 前30日の平均Mが3.0〜3.5で発生回数が急増（M7.5前兆パターン）
        prev_30_data = df[(df['date'] >= day - pd.Timedelta(days=30)) & (df['date'] < day)]
        if len(prev_30_data) > 0:
            avg_m = prev_30_data['magnitude'].mean()
            count_30 = len(prev_30_data)
            count_today = len(daily_data)
            
            # 平均Mが3.0〜3.5の範囲で、発生回数が急増している場合
            if 3.0 <= avg_m <= 3.5 and count_today >= count_30 / 30 * 2:
                daily_scores.loc[day, 'score'] += 1.5
        
        # M7.5前兆強化: 前震候補スコア（M7.5発生前30日以内でM5〜M6.9が多発）
        if any((m75_date - pd.Timedelta(days=30) <= day < m75_date) for m75_date in m75_dates):
            m5_to_m69_count = daily_data[(daily_data['magnitude'] >= 5) & (daily_data['magnitude'] < 7)].shape[0]
            if m5_to_m69_count >= 3:
                daily_scores.loc[day, 'score'] += 2.0  # M7.5前兆として重みを増加
            elif m5_to_m69_count >= 2:
                daily_scores.loc[day, 'score'] += 1.0
        
        # M7前兆: 前震候補スコア（M7発生前30日以内でM5〜M6.9が多発）
        elif any((m7_date - pd.Timedelta(days=30) <= day < m7_date) for m7_date in m7_dates):
            if daily_data[(daily_data['magnitude'] >= 5) & (daily_data['magnitude'] < 7)].shape[0] >= 3:
                daily_scores.loc[day, 'score'] += 1.5
    
    return daily_scores


def calculate_score_world(df, base_date):
    """
    世界地震スコアを計算する（M7.5前兆強化版）
    
    Parameters:
    -----------
    df : pd.DataFrame
        世界地震データ
    base_date : str
        基準日（YYYY-MM-DD形式）
    
    Returns:
    --------
    float
        スコア
    """
    end_date = pd.to_datetime(base_date)
    start_date = end_date - pd.DateOffset(days=30)
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    score = 0
    
    # M7.5以上の前兆検出を強化
    for m in filtered_df['magnitude_w']:
        if m >= 6:
            score += 1/20
        if m >= 7:
            score += 2/20
        if m >= 7.5:  # M7.5以上は重みを増加
            score += 2/20  # 追加スコア
        if m >= 8:
            score += 3/20
    
    # 世界M7発生後7日間は国内地震リスクが高まる（M7.5前兆パターン）
    # 過去7日以内に世界M7以上が発生している場合、スコアを加算
    recent_m7 = df[(df['date'] >= end_date - pd.DateOffset(days=7)) & 
                   (df['date'] <= end_date) & 
                   (df['magnitude_w'] >= 7)]
    if len(recent_m7) > 0:
        score += 1.0  # 世界M7発生後の国内地震リスク加算
    
    return score


def calculate_score_depth(df, base_date):
    """
    深さスコアを計算する
    
    Parameters:
    -----------
    df : pd.DataFrame
        深さデータ
    base_date : str
        基準日（YYYY-MM-DD形式）
    
    Returns:
    --------
    float
        スコア
    """
    end_date = pd.to_datetime(base_date)
    start_date = end_date - pd.DateOffset(days=30)
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    score = 0
    for depth in filtered_df['depth']:
        if depth >= 300:
            score += 1/20
        if depth >= 400:
            score += 2/20
        if depth >= 500:
            score += 3/20
    return score


def calculate_daily_displacement(df):
    """
    日々の変動量を計算（前日からの変化量）
    基準点からの累積変動量ではなく、日々の変動量を計算する
    """
    df = df.sort_values(['station_id', 'date']).copy()
    
    # 各地域ごとに前日からの変化量を計算
    df['prev_east'] = df.groupby('station_id')['east_mm'].shift(1)
    df['prev_north'] = df.groupby('station_id')['north_mm'].shift(1)
    df['prev_up'] = df.groupby('station_id')['up_mm'].shift(1)
    
    # 日々の変動量（前日からの変化量の絶対値）
    df['daily_east_change'] = (df['east_mm'] - df['prev_east']).abs()
    df['daily_north_change'] = (df['north_mm'] - df['prev_north']).abs()
    df['daily_up_change'] = (df['up_mm'] - df['prev_up']).abs()
    
    # 日々の総変動量
    df['daily_displacement'] = (
        df['daily_east_change'] + 
        df['daily_north_change'] + 
        df['daily_up_change']
    )
    
    # 最初の日（基準日）は変動量0とする
    df.loc[df['daily_displacement'].isna(), 'daily_displacement'] = 0
    
    return df


def calculate_gnss_score(df):
    """
    GPSデータに基づいて地殻変動スコアを計算
    
    Parameters:
    -----------
    df : pd.DataFrame
        GNSSデータ（date, station_id, east_mm, north_mm, up_mmカラムが必要）
    
    Returns:
    --------
    pd.DataFrame
        日毎のスコア（date, gnss_scoreカラム）
    """
    print("  日々の変動量を計算中...")
    # 日々の変動量を計算
    df = calculate_daily_displacement(df)
    
    print("  連続変動日数を計算中...")
    # 閾値: 10mm以上の変動を「変動あり」とする
    threshold = 10.0
    df['has_displacement'] = df['daily_displacement'] >= threshold
    
    # 各地域の連続変動日数を計算
    df['consecutive_days'] = 0
    for station_id in df['station_id'].unique():
        station_mask = df['station_id'] == station_id
        station_data = df[station_mask].sort_values('date').copy()
        
        # 連続日数を計算
        station_data['consecutive_days'] = (
            station_data['has_displacement']
            .groupby((station_data['has_displacement'] != station_data['has_displacement'].shift()).cumsum())
            .transform('size')
        )
        
        # 元のDataFrameに反映
        df.loc[station_mask, 'consecutive_days'] = station_data['consecutive_days'].values
    
    print("  日付ごとのスコアを計算中...")
    # 日付ごとにスコアを計算
    dates = sorted(df['date'].unique())
    results = []
    total_dates = len(dates)
    
    for idx, date in enumerate(dates, 1):
        if idx % 100 == 0:
            print(f"    進捗: {idx}/{total_dates}日 ({idx*100//total_dates}%)")
        
        date_data = df[df['date'] == date].copy()
        
        score = 0.0
        
        # 1. 変動量スコア（各地域の最大変動量を評価）
        max_displacement = date_data['daily_displacement'].max()
        if max_displacement >= 50:
            score += 2.0
        elif max_displacement >= 20:
            score += 1.0
        elif max_displacement >= 10:
            score += 0.5
        
        # 2. 継続性スコア（各地域の連続変動日数を評価）
        for _, row in date_data.iterrows():
            if row['has_displacement']:
                consecutive_days = row['consecutive_days']
                
                if consecutive_days >= 14:
                    score += 2.0
                elif consecutive_days >= 7:
                    score += 1.0
                elif consecutive_days >= 3:
                    score += 0.5
        
        # 3. 同時変動スコア（複数地点で同時に変動が発生している場合）
        num_simultaneous = date_data['has_displacement'].sum()
        
        if num_simultaneous >= 5:
            score += 2.0
        elif num_simultaneous >= 3:
            score += 1.0
        
        results.append({
            'date': date.strftime('%Y-%m-%d'),
            'gnss_score': score
        })
    
    return pd.DataFrame(results)


def main():
    """メイン処理"""
    import sys

    # コマンドライン引数の処理
    include_gnss = '--no-gnss' not in sys.argv

    # ファイル名のサフィックス（GNSSあり/なしを区別）
    suffix = "_with_gnss" if include_gnss else "_no_gnss"

    print("=" * 60)
    print("地震データ分析システム - スコア計算開始")
    print("=" * 60)
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"生データフォルダ: {RAW_DATA_DIR}")
    print(f"出力先フォルダ: {OUTPUT_DIR}")
    if not include_gnss:
        print("モード: GNSSスコアなし（--no-gnss）")
        print("ファイル名サフィックス: _no_gnss")
    else:
        print("モード: GNSSスコアあり")
        print("ファイル名サフィックス: _with_gnss")
    print("-" * 60)

    # 1. 国内地震スコア計算
    print("\n[1/6] 国内地震スコアを計算中...")
    try:
        japan_file = RAW_DATA_DIR / "2010-all.csv"
        if not japan_file.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {japan_file}")
        
        df_japan = pd.read_csv(japan_file)
        df_japan['date'] = df_japan['date'].apply(parse_japan_earthquake_date)
        df_japan['magnitude'] = pd.to_numeric(df_japan['magnitude'], errors='coerce')
        
        daily_scores = calculate_daily_score(df_japan)
        output_file = OUTPUT_DIR / "japan_scores.csv"
        daily_scores.to_csv(output_file, index=True)
        print(f"[OK] 出力完了: {output_file}")
        print(f"  データ件数: {len(daily_scores)}件")
    except Exception as e:
        print(f"[NG] エラー: {e}")
        return
    
    # 2. 世界地震スコア計算
    print("\n[2/6] 世界地震スコアを計算中...")
    try:
        world_file = RAW_DATA_DIR / "world-all.csv"
        if not world_file.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {world_file}")
        
        dfw = pd.read_csv(world_file, parse_dates=['date'])
        results_world = []
        for base_date in pd.date_range(pd.to_datetime('2010-02-01'), dfw['date'].max()):
            score = calculate_score_world(dfw, base_date.strftime('%Y-%m-%d'))
            results_world.append([base_date.strftime("%Y-%m-%d"), score])
        
        output_file = OUTPUT_DIR / "world_scores.csv"
        pd.DataFrame(results_world, columns=['date', 'world_score']).to_csv(output_file, index=False)
        print(f"[OK] 出力完了: {output_file}")
        print(f"  データ件数: {len(results_world)}件")
    except Exception as e:
        print(f"[NG] エラー: {e}")
        return
    
    # 3. 深さスコア計算
    print("\n[3/6] 深さスコアを計算中...")
    try:
        depth_file = RAW_DATA_DIR / "depth.csv"
        if not depth_file.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {depth_file}")
        
        dfd = pd.read_csv(depth_file, parse_dates=['date'])
        results_depth = []
        for base_date in pd.date_range(pd.to_datetime('2010-02-01'), dfd['date'].max()):
            score = calculate_score_depth(dfd, base_date.strftime('%Y-%m-%d'))
            results_depth.append([base_date.strftime("%Y-%m-%d"), score])
        
        output_file = OUTPUT_DIR / "depth_scores.csv"
        pd.DataFrame(results_depth, columns=['date', 'depth_score']).to_csv(output_file, index=False)
        print(f"[OK] 出力完了: {output_file}")
        print(f"  データ件数: {len(results_depth)}件")
    except Exception as e:
        print(f"[NG] エラー: {e}")
        return
    
    # 4. GNSSスコア計算
    gnss_scores = None
    if include_gnss:
        print("\n[4/6] GNSSスコアを計算中...")
        try:
            gnss_file = RAW_DATA_DIR / "gnss_data.csv"
            if not gnss_file.exists():
                print(f"[!] 警告: GNSSデータファイルが見つかりません: {gnss_file}")
                print("  GNSSスコアはスキップします")
            else:
                df_gnss = pd.read_csv(gnss_file)
                df_gnss['date'] = pd.to_datetime(df_gnss['date'], format='%Y/%m/%d')

                # GNSSスコア計算関数を呼び出し
                gnss_scores = calculate_gnss_score(df_gnss)
                output_file = OUTPUT_DIR / "gnss_scores.csv"
                gnss_scores.to_csv(output_file, index=False)
                print(f"[OK] 出力完了: {output_file}")
                print(f"  データ件数: {len(gnss_scores)}件")
                print(f"  最大スコア: {gnss_scores['gnss_score'].max():.2f}")
                print(f"  平均スコア: {gnss_scores['gnss_score'].mean():.2f}")
        except Exception as e:
            print(f"[NG] エラー: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[4/6] GNSSスコア計算をスキップ（--no-gnss モード）")

    # 5. b値スコア計算
    print("\n[5/6] b値スコアを計算中...")
    b_value_scores = None
    try:
        # まずベースラインを計算
        print("  ベースラインb値を計算中...")
        baseline_b, baseline_std = calculate_baseline_b_value(df_japan, mc=2.0, window_days=30)
        if baseline_b is not None:
            print(f"  ベースラインb値: {baseline_b:.3f} (標準偏差: {baseline_std:.3f})")
        else:
            print("  [!] ベースラインb値を計算できませんでした")

        # 国内地震データを使用してb値スコアを計算
        dates = sorted(df_japan['date'].unique())
        results_b = []
        total_dates = len(dates)

        for idx, current_date in enumerate(dates, 1):
            if idx % 500 == 0:
                print(f"  進捗: {idx}/{total_dates}日 ({idx*100//total_dates}%)")

            score, b_val = calculate_b_value_score(
                df_japan, current_date, window_days=30, mc=2.0,
                baseline_b=baseline_b, baseline_std=baseline_std
            )
            results_b.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'b_value': b_val,
                'b_value_score': score
            })

        b_value_scores = pd.DataFrame(results_b)
        output_file = OUTPUT_DIR / "b_value_scores.csv"
        b_value_scores.to_csv(output_file, index=False)
        print(f"[OK] 出力完了: {output_file}")
        print(f"  データ件数: {len(b_value_scores)}件")

        # b値の統計情報を表示
        valid_b = b_value_scores['b_value'].dropna()
        if len(valid_b) > 0:
            print(f"  b値の範囲: {valid_b.min():.3f} - {valid_b.max():.3f}")
            print(f"  b値の平均: {valid_b.mean():.3f}")
            # スコアが付いた日数を表示
            scored_days = len(b_value_scores[b_value_scores['b_value_score'] > 0])
            print(f"  b値スコア > 0 の日数: {scored_days}日")
    except Exception as e:
        print(f"[NG] エラー: {e}")
        import traceback
        traceback.print_exc()

    # 6. 総合スコア計算
    print("\n[6/6] 総合スコアを計算中...")
    try:
        # 各スコアファイルを読み込み
        depth_scores = pd.read_csv(OUTPUT_DIR / "depth_scores.csv", parse_dates=['date'])
        world_scores = pd.read_csv(OUTPUT_DIR / "world_scores.csv", parse_dates=['date'])
        japan_scores = pd.read_csv(OUTPUT_DIR / "japan_scores.csv", parse_dates=['date']).rename(columns={'score': 'japan_score'})
        
        # マージ処理
        merged_df = pd.merge(depth_scores, world_scores, on='date', how='outer')
        merged_df = pd.merge(merged_df, japan_scores, on='date', how='outer')
        
        # GNSSスコアをマージ（存在する場合）
        if gnss_scores is not None:
            gnss_scores_parsed = pd.read_csv(OUTPUT_DIR / "gnss_scores.csv", parse_dates=['date'])
            merged_df = pd.merge(merged_df, gnss_scores_parsed, on='date', how='outer')
        else:
            merged_df['gnss_score'] = 0

        # b値スコアをマージ（存在する場合）
        if b_value_scores is not None:
            b_scores_parsed = pd.read_csv(OUTPUT_DIR / "b_value_scores.csv", parse_dates=['date'])
            merged_df = pd.merge(merged_df, b_scores_parsed[['date', 'b_value_score']], on='date', how='outer')
        else:
            merged_df['b_value_score'] = 0

        merged_df = merged_df.fillna(0)

        # 総合スコア計算（国内+世界+深さ+GNSS+b値）
        merged_df['total_score'] = (
            merged_df['depth_score'] +
            merged_df['world_score'] +
            merged_df['japan_score'] +
            merged_df['gnss_score'] +
            merged_df['b_value_score']
        )
        
        output_file = OUTPUT_DIR / f"total{suffix}.csv"
        merged_df[['date', 'total_score']].to_csv(output_file, index=False)
        print(f"[OK] 出力完了: {output_file}")
        print(f"  データ件数: {len(merged_df)}件")
        print(f"  最大スコア: {merged_df['total_score'].max():.2f}")
        print(f"  平均スコア: {merged_df['total_score'].mean():.2f}")
    except Exception as e:
        print(f"[NG] エラー: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("全ての処理が完了しました！")
    print("=" * 60)
    print(f"\n出力ファイル:")
    print(f"  - {OUTPUT_DIR / 'japan_scores.csv'}")
    print(f"  - {OUTPUT_DIR / 'world_scores.csv'}")
    print(f"  - {OUTPUT_DIR / 'depth_scores.csv'}")
    print(f"  - {OUTPUT_DIR / 'b_value_scores.csv'}")
    if gnss_scores is not None:
        print(f"  - {OUTPUT_DIR / 'gnss_scores.csv'}")
        print(f"  - {OUTPUT_DIR / f'total{suffix}.csv'} (総合スコア: 国内+世界+深さ+GNSS+b値)")
    else:
        print(f"  - {OUTPUT_DIR / f'total{suffix}.csv'} (総合スコア: 国内+世界+深さ+b値、GNSSなし)")


if __name__ == "__main__":
    main()

