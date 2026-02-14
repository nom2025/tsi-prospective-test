"""
地震データ特徴探索スクリプト

生データから新しい特徴やパターンを探索し、分析結果を出力します。
既存のスコア計算で使用されていない特徴を発見することを目的とします。

実行方法:
    python ソース/explore_features.py
"""

import sys
import importlib.util
from pathlib import Path

# 必要なパッケージのチェック
REQUIRED_PACKAGES = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'scipy': 'scipy'
}

def check_package(package_name, import_name):
    """パッケージがインストールされているかチェック"""
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            return False
        # 実際にインポートを試みる
        __import__(import_name)
        return True
    except (ImportError, ModuleNotFoundError):
        return False

missing_packages = []
for package_name, import_name in REQUIRED_PACKAGES.items():
    if not check_package(package_name, import_name):
        missing_packages.append(package_name)

if missing_packages:
    print("=" * 60)
    print("エラー: 必要なパッケージがインストールされていません")
    print("=" * 60)
    print(f"\n不足しているパッケージ: {', '.join(missing_packages)}")
    print(f"\n現在のPython実行環境: {sys.executable}")
    print("\n以下のコマンドでインストールしてください:")
    print(f"  {sys.executable} -m pip install {' '.join(missing_packages)}")
    print("\nまたは、requirements.txtから全てインストール:")
    print(f"  {sys.executable} -m pip install -r requirements.txt")
    print("\n注意: 仮想環境を使用している場合は、")
    print("      その環境でパッケージをインストールしてください。")
    print("=" * 60)
    sys.exit(1)

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートのパスを取得
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "生データ"
OUTPUT_DIR = PROJECT_ROOT / "データ分析結果出力" / "特徴探索結果"
GRAPH_DIR = PROJECT_ROOT / "グラフ"

# 出力ディレクトリが存在しない場合は作成
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# 日本語フォント設定
japanese_fonts = ['MS Gothic', 'MS PGothic', 'Yu Gothic', 'Meiryo', 'メイリオ', 'MS ゴシック']
font_found = False
for font in japanese_fonts:
    try:
        plt.rcParams['font.family'] = font
        font_found = True
        break
    except:
        continue
if not font_found:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def parse_japan_earthquake_date(date_str):
    """国内地震データの日付形式を統一する"""
    if '月' in str(date_str) and '日' in str(date_str) and '年' not in str(date_str):
        return pd.to_datetime(f'2010年{date_str}', format='%Y年%m月%d日')
    else:
        return pd.to_datetime(date_str, format='%Y/%m/%d')


def explore_regional_patterns(df_japan):
    """
    地域別のパターンを分析
    - 地域ごとのマグニチュード分布
    - 地域間の相関
    - 地域別の発生頻度変化
    """
    print("\n[特徴1] 地域別パターン分析")
    print("-" * 60)
    
    # 地域ごとの統計
    regional_stats = df_japan.groupby('place').agg({
        'magnitude': ['count', 'mean', 'std', 'min', 'max'],
        'date': ['min', 'max']
    }).round(2)
    regional_stats.columns = ['発生回数', '平均M', '標準偏差', '最小M', '最大M', '初回発生', '最終発生']
    regional_stats = regional_stats.sort_values('発生回数', ascending=False)
    
    # 上位20地域を出力
    print("\n発生回数上位20地域:")
    print(regional_stats.head(20).to_string())
    
    # 地域別のマグニチュード分布の特徴
    high_freq_regions = regional_stats.head(10).index.tolist()
    
    # 地域間の相関（日毎の発生回数）
    regional_daily = df_japan.groupby(['date', 'place']).size().unstack(fill_value=0)
    regional_corr = regional_daily.corr()
    
    # 強い相関を持つ地域ペアを抽出（相関係数 > 0.3）
    high_corr_pairs = []
    for i in range(len(regional_corr.columns)):
        for j in range(i+1, len(regional_corr.columns)):
            corr_val = regional_corr.iloc[i, j]
            if corr_val > 0.3:
                high_corr_pairs.append({
                    '地域1': regional_corr.columns[i],
                    '地域2': regional_corr.columns[j],
                    '相関係数': corr_val
                })
    
    if high_corr_pairs:
        print(f"\n強い相関を持つ地域ペア（相関係数 > 0.3）: {len(high_corr_pairs)}組")
        corr_df = pd.DataFrame(high_corr_pairs).sort_values('相関係数', ascending=False)
        print(corr_df.head(10).to_string(index=False))
    
    # CSV出力
    regional_stats.to_csv(OUTPUT_DIR / "regional_statistics.csv", encoding='utf-8-sig')
    if high_corr_pairs:
        corr_df.to_csv(OUTPUT_DIR / "regional_correlations.csv", index=False, encoding='utf-8-sig')
    
    return regional_stats, high_corr_pairs


def explore_magnitude_trends(df_japan):
    """
    マグニチュードの時系列トレンド分析
    - 移動平均によるトレンド
    - マグニチュードの変化率
    - 高マグニチュード地震の前兆パターン
    """
    print("\n[特徴2] マグニチュードトレンド分析")
    print("-" * 60)
    
    # 日毎の統計
    daily_stats = df_japan.groupby('date').agg({
        'magnitude': ['mean', 'std', 'max', 'count']
    })
    daily_stats.columns = ['平均M', '標準偏差', '最大M', '発生回数']
    
    # 7日移動平均
    daily_stats['平均M_7日移動平均'] = daily_stats['平均M'].rolling(window=7, center=True).mean()
    daily_stats['最大M_7日移動平均'] = daily_stats['最大M'].rolling(window=7, center=True).mean()
    
    # 30日移動平均
    daily_stats['平均M_30日移動平均'] = daily_stats['平均M'].rolling(window=30, center=True).mean()
    
    # 変化率（前日比）
    daily_stats['平均M_変化率'] = daily_stats['平均M'].pct_change()
    daily_stats['最大M_変化率'] = daily_stats['最大M'].pct_change()
    
    # M7以上の地震発生前のパターン
    m7_dates = df_japan[df_japan['magnitude'] >= 7]['date'].unique()
    
    pre_m7_patterns = []
    for m7_date in m7_dates:
        pre_30_days = daily_stats[
            (daily_stats.index >= m7_date - pd.Timedelta(days=30)) &
            (daily_stats.index < m7_date)
        ]
        if len(pre_30_days) > 0:
            pre_m7_patterns.append({
                'M7発生日': m7_date,
                '前30日平均M': pre_30_days['平均M'].mean(),
                '前30日最大M平均': pre_30_days['最大M'].mean(),
                '前30日発生回数平均': pre_30_days['発生回数'].mean(),
                '前30日平均M標準偏差': pre_30_days['平均M'].std(),
            })
    
    if pre_m7_patterns:
        pre_m7_df = pd.DataFrame(pre_m7_patterns)
        print("\nM7以上地震発生前30日の特徴:")
        print(pre_m7_df.to_string(index=False))
        pre_m7_df.to_csv(OUTPUT_DIR / "pre_m7_patterns.csv", index=False, encoding='utf-8-sig')
    
    # CSV出力
    daily_stats.to_csv(OUTPUT_DIR / "magnitude_trends.csv", encoding='utf-8-sig')
    
    return daily_stats


def explore_time_intervals(df_japan):
    """
    地震発生間隔の分析
    - 地震間隔の分布
    - 短い間隔での連続発生パターン
    """
    print("\n[特徴3] 地震発生間隔分析")
    print("-" * 60)
    
    # 日付でソート
    df_sorted = df_japan.sort_values('date')
    
    # 日毎の発生回数
    daily_counts = df_sorted.groupby('date').size()
    
    # 地震発生間隔（日単位）
    date_intervals = daily_counts.index.to_series().diff().dt.days.dropna()
    
    # 統計
    interval_stats = {
        '平均間隔': date_intervals.mean(),
        '中央値間隔': date_intervals.median(),
        '最小間隔': date_intervals.min(),
        '最大間隔': date_intervals.max(),
        '標準偏差': date_intervals.std(),
    }
    
    print("\n地震発生間隔の統計（日単位）:")
    for key, value in interval_stats.items():
        print(f"  {key}: {value:.2f}日")
    
    # 短い間隔（1日以内）での連続発生
    consecutive_days = []
    current_streak = 0
    prev_date = None
    
    for date in sorted(daily_counts.index):
        if prev_date is None:
            current_streak = 1
        elif (date - prev_date).days <= 1:
            current_streak += 1
        else:
            if current_streak >= 3:
                consecutive_days.append({
                    '開始日': prev_date - pd.Timedelta(days=current_streak-1),
                    '終了日': prev_date,
                    '連続日数': current_streak
                })
            current_streak = 1
        prev_date = date
    
    if consecutive_days:
        consecutive_df = pd.DataFrame(consecutive_days)
        print(f"\n3日以上連続発生した期間: {len(consecutive_days)}回")
        print(consecutive_df.head(10).to_string(index=False))
        consecutive_df.to_csv(OUTPUT_DIR / "consecutive_periods.csv", index=False, encoding='utf-8-sig')
    
    return date_intervals, interval_stats


def explore_world_japan_correlation(df_world, df_japan):
    """
    世界地震と国内地震の相関分析
    - 世界地震発生後の国内地震への影響
    - 時間差相関
    """
    print("\n[特徴4] 世界地震と国内地震の相関分析")
    print("-" * 60)
    
    # 日毎の世界地震統計（M6以上）
    world_daily = df_world[df_world['magnitude_w'] >= 6].groupby('date').agg({
        'magnitude_w': ['count', 'max', 'mean']
    })
    world_daily.columns = ['世界M6以上回数', '世界最大M', '世界平均M']
    
    # 日毎の国内地震統計
    japan_daily = df_japan.groupby('date').agg({
        'magnitude': ['count', 'max', 'mean']
    })
    japan_daily.columns = ['国内発生回数', '国内最大M', '国内平均M']
    
    # マージ
    merged = pd.merge(world_daily, japan_daily, left_index=True, right_index=True, how='outer').fillna(0)
    
    # 世界地震発生後の国内地震への影響（1-7日後）
    correlations = {}
    for lag in range(1, 8):
        world_shifted = merged['世界M6以上回数'].shift(lag)
        corr = world_shifted.corr(merged['国内発生回数'])
        correlations[f'{lag}日後'] = corr
    
    print("\n世界地震発生後の国内地震への相関（時間差相関）:")
    for lag, corr in correlations.items():
        print(f"  {lag}: {corr:.4f}")
    
    # M7以上の世界地震発生後の国内地震パターン
    world_m7_dates = df_world[df_world['magnitude_w'] >= 7]['date'].unique()
    
    post_world_m7_patterns = []
    for world_m7_date in world_m7_dates:
        post_7_days = merged[
            (merged.index > world_m7_date) &
            (merged.index <= world_m7_date + pd.Timedelta(days=7))
        ]
        if len(post_7_days) > 0:
            post_world_m7_patterns.append({
                '世界M7発生日': world_m7_date,
                '7日後国内発生回数平均': post_7_days['国内発生回数'].mean(),
                '7日後国内最大M平均': post_7_days['国内最大M'].mean(),
                '7日後国内平均M': post_7_days['国内平均M'].mean(),
            })
    
    if post_world_m7_patterns:
        post_world_m7_df = pd.DataFrame(post_world_m7_patterns)
        print("\n世界M7以上地震発生後7日間の国内地震特徴:")
        print(post_world_m7_df.to_string(index=False))
        post_world_m7_df.to_csv(OUTPUT_DIR / "post_world_m7_patterns.csv", index=False, encoding='utf-8-sig')
    
    merged.to_csv(OUTPUT_DIR / "world_japan_correlation.csv", encoding='utf-8-sig')
    
    return merged, correlations


def explore_depth_magnitude_relationship(df_depth, df_japan):
    """
    深さとマグニチュードの関係分析
    - 深さデータと国内地震のマグニチュードの相関
    """
    print("\n[特徴5] 深さとマグニチュードの関係分析")
    print("-" * 60)
    
    # 深さデータを日毎に集計
    depth_daily = df_depth.groupby('date').agg({
        'depth': ['mean', 'max', 'min', 'count']
    })
    depth_daily.columns = ['平均深さ', '最大深さ', '最小深さ', '深さデータ数']
    
    # 国内地震を日毎に集計
    japan_daily = df_japan.groupby('date').agg({
        'magnitude': ['mean', 'max', 'count']
    })
    japan_daily.columns = ['国内平均M', '国内最大M', '国内発生回数']
    
    # マージ（±3日以内のデータをマッチング）
    merged_list = []
    for depth_date in depth_daily.index:
        # ±3日以内の国内地震データを探す
        nearby_japan = japan_daily[
            (japan_daily.index >= depth_date - pd.Timedelta(days=3)) &
            (japan_daily.index <= depth_date + pd.Timedelta(days=3))
        ]
        if len(nearby_japan) > 0:
            merged_list.append({
                'date': depth_date,
                '平均深さ': depth_daily.loc[depth_date, '平均深さ'],
                '最大深さ': depth_daily.loc[depth_date, '最大深さ'],
                '深さデータ数': depth_daily.loc[depth_date, '深さデータ数'],
                '国内平均M': nearby_japan['国内平均M'].mean(),
                '国内最大M': nearby_japan['国内最大M'].max(),
                '国内発生回数': nearby_japan['国内発生回数'].sum(),
            })
    
    if merged_list:
        merged = pd.DataFrame(merged_list)
        
        # 相関分析
        correlations = {
            '平均深さ vs 国内平均M': merged['平均深さ'].corr(merged['国内平均M']),
            '最大深さ vs 国内最大M': merged['最大深さ'].corr(merged['国内最大M']),
            '平均深さ vs 国内発生回数': merged['平均深さ'].corr(merged['国内発生回数']),
        }
        
        print("\n深さと国内地震の相関:")
        for key, value in correlations.items():
            print(f"  {key}: {value:.4f}")
        
        # 深さ別のマグニチュード分布
        merged['深さカテゴリ'] = pd.cut(merged['平均深さ'], 
                                      bins=[0, 300, 400, 500, float('inf')],
                                      labels=['浅い(<300)', '中(300-400)', '深い(400-500)', '非常に深い(>500)'])
        
        depth_category_stats = merged.groupby('深さカテゴリ').agg({
            '国内平均M': 'mean',
            '国内最大M': 'mean',
            '国内発生回数': 'mean'
        }).round(2)
        
        print("\n深さカテゴリ別の国内地震特徴:")
        print(depth_category_stats.to_string())
        
        merged.to_csv(OUTPUT_DIR / "depth_magnitude_relationship.csv", index=False, encoding='utf-8-sig')
        depth_category_stats.to_csv(OUTPUT_DIR / "depth_category_statistics.csv", encoding='utf-8-sig')
        
        return merged, correlations
    
    return None, None


def explore_seasonal_patterns(df_japan):
    """
    季節性パターンの分析
    - 月別の発生回数・マグニチュード
    - 季節による違い
    """
    print("\n[特徴6] 季節性パターン分析")
    print("-" * 60)
    
    df_japan['月'] = df_japan['date'].dt.month
    df_japan['季節'] = df_japan['date'].dt.month.map({
        12: '冬', 1: '冬', 2: '冬',
        3: '春', 4: '春', 5: '春',
        6: '夏', 7: '夏', 8: '夏',
        9: '秋', 10: '秋', 11: '秋'
    })
    
    # 月別統計
    monthly_stats = df_japan.groupby('月').agg({
        'magnitude': ['count', 'mean', 'max'],
        'date': 'nunique'
    })
    monthly_stats.columns = ['発生回数', '平均M', '最大M', '発生日数']
    monthly_stats['1日あたり発生回数'] = (monthly_stats['発生回数'] / monthly_stats['発生日数']).round(2)
    
    print("\n月別統計:")
    print(monthly_stats.to_string())
    
    # 季節別統計
    seasonal_stats = df_japan.groupby('季節').agg({
        'magnitude': ['count', 'mean', 'max'],
        'date': 'nunique'
    })
    seasonal_stats.columns = ['発生回数', '平均M', '最大M', '発生日数']
    seasonal_stats['1日あたり発生回数'] = (seasonal_stats['発生回数'] / seasonal_stats['発生日数']).round(2)
    
    print("\n季節別統計:")
    print(seasonal_stats.to_string())
    
    monthly_stats.to_csv(OUTPUT_DIR / "monthly_statistics.csv", encoding='utf-8-sig')
    seasonal_stats.to_csv(OUTPUT_DIR / "seasonal_statistics.csv", encoding='utf-8-sig')
    
    return monthly_stats, seasonal_stats


def explore_magnitude_distribution_features(df_japan):
    """
    マグニチュード分布の特徴量
    - 歪度、尖度
    - 分布の形状変化
    """
    print("\n[特徴7] マグニチュード分布の特徴量分析")
    print("-" * 60)
    
    from scipy import stats
    
    # 日毎の分布特徴量
    daily_dist_features = []
    
    for date in sorted(df_japan['date'].unique()):
        daily_mags = df_japan[df_japan['date'] == date]['magnitude'].values
        
        if len(daily_mags) >= 3:  # 最低3つのデータが必要
            daily_dist_features.append({
                'date': date,
                '発生回数': len(daily_mags),
                '平均M': np.mean(daily_mags),
                '中央値M': np.median(daily_mags),
                '標準偏差': np.std(daily_mags),
                '歪度': stats.skew(daily_mags),
                '尖度': stats.kurtosis(daily_mags),
                '最小M': np.min(daily_mags),
                '最大M': np.max(daily_mags),
                '25%分位': np.percentile(daily_mags, 25),
                '75%分位': np.percentile(daily_mags, 75),
            })
    
    if daily_dist_features:
        dist_features_df = pd.DataFrame(daily_dist_features)
        
        print("\n分布特徴量の統計:")
        print(dist_features_df[['発生回数', '平均M', '標準偏差', '歪度', '尖度']].describe().round(2))
        
        # M7発生前の分布特徴
        m7_dates = df_japan[df_japan['magnitude'] >= 7]['date'].unique()
        
        pre_m7_dist_features = []
        for m7_date in m7_dates:
            pre_30 = dist_features_df[
                (dist_features_df['date'] >= m7_date - pd.Timedelta(days=30)) &
                (dist_features_df['date'] < m7_date)
            ]
            if len(pre_30) > 0:
                pre_m7_dist_features.append({
                    'M7発生日': m7_date,
                    '前30日平均歪度': pre_30['歪度'].mean(),
                    '前30日平均尖度': pre_30['尖度'].mean(),
                    '前30日標準偏差平均': pre_30['標準偏差'].mean(),
                })
        
        if pre_m7_dist_features:
            pre_m7_dist_df = pd.DataFrame(pre_m7_dist_features)
            print("\nM7発生前30日の分布特徴:")
            print(pre_m7_dist_df.to_string(index=False))
            pre_m7_dist_df.to_csv(OUTPUT_DIR / "pre_m7_distribution_features.csv", index=False, encoding='utf-8-sig')
        
        dist_features_df.to_csv(OUTPUT_DIR / "magnitude_distribution_features.csv", index=False, encoding='utf-8-sig')
        
        return dist_features_df
    
    return None


def create_feature_summary_report():
    """発見した特徴のサマリーレポートを作成"""
    print("\n" + "=" * 60)
    print("特徴探索サマリーレポート")
    print("=" * 60)
    
    report_lines = [
        "# 地震データ特徴探索レポート",
        f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 発見された特徴",
        "",
        "### 1. 地域別パターン",
        "- 地域ごとの発生頻度、マグニチュード分布",
        "- 地域間の相関関係",
        "- 出力ファイル: regional_statistics.csv, regional_correlations.csv",
        "",
        "### 2. マグニチュードトレンド",
        "- 移動平均によるトレンド分析",
        "- M7発生前のパターン",
        "- 出力ファイル: magnitude_trends.csv, pre_m7_patterns.csv",
        "",
        "### 3. 地震発生間隔",
        "- 地震発生間隔の分布",
        "- 連続発生パターン",
        "- 出力ファイル: consecutive_periods.csv",
        "",
        "### 4. 世界地震と国内地震の相関",
        "- 時間差相関分析",
        "- 世界M7発生後の国内地震パターン",
        "- 出力ファイル: world_japan_correlation.csv, post_world_m7_patterns.csv",
        "",
        "### 5. 深さとマグニチュードの関係",
        "- 深さと国内地震の相関",
        "- 深さカテゴリ別の特徴",
        "- 出力ファイル: depth_magnitude_relationship.csv, depth_category_statistics.csv",
        "",
        "### 6. 季節性パターン",
        "- 月別・季節別の統計",
        "- 出力ファイル: monthly_statistics.csv, seasonal_statistics.csv",
        "",
        "### 7. マグニチュード分布の特徴量",
        "- 歪度、尖度などの分布形状指標",
        "- M7発生前の分布特徴",
        "- 出力ファイル: magnitude_distribution_features.csv, pre_m7_distribution_features.csv",
        "",
        "## 次のステップ",
        "- これらの特徴をスコア計算に組み込む",
        "- 機械学習モデルでの特徴量として活用",
        "- 可視化によるパターンの確認",
    ]
    
    report_path = OUTPUT_DIR / "feature_exploration_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print("\n".join(report_lines))
    print(f"\nレポートを保存しました: {report_path}")


def main():
    """メイン処理"""
    print("=" * 60)
    print("地震データ特徴探索スクリプト")
    print("=" * 60)
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"生データフォルダ: {RAW_DATA_DIR}")
    print(f"出力先フォルダ: {OUTPUT_DIR}")
    print("-" * 60)
    
    # データ読み込み
    print("\n[データ読み込み]")
    try:
        # 国内地震データ
        japan_file = RAW_DATA_DIR / "2010-all.csv"
        df_japan = pd.read_csv(japan_file)
        df_japan['date'] = df_japan['date'].apply(parse_japan_earthquake_date)
        df_japan['magnitude'] = pd.to_numeric(df_japan['magnitude'], errors='coerce')
        df_japan = df_japan.dropna(subset=['date', 'magnitude'])
        print(f"✓ 国内地震データ: {len(df_japan)}件")
        
        # 世界地震データ
        world_file = RAW_DATA_DIR / "world-all.csv"
        df_world = pd.read_csv(world_file, parse_dates=['date'])
        df_world['magnitude_w'] = pd.to_numeric(df_world['magnitude_w'], errors='coerce')
        df_world = df_world.dropna(subset=['date', 'magnitude_w'])
        print(f"✓ 世界地震データ: {len(df_world)}件")
        
        # 深さデータ
        depth_file = RAW_DATA_DIR / "depth.csv"
        df_depth = pd.read_csv(depth_file, parse_dates=['date'])
        df_depth['depth'] = pd.to_numeric(df_depth['depth'], errors='coerce')
        df_depth = df_depth.dropna(subset=['date', 'depth'])
        print(f"✓ 深さデータ: {len(df_depth)}件")
        
    except Exception as e:
        print(f"✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 特徴探索
    try:
        # 1. 地域別パターン
        regional_stats, regional_corr = explore_regional_patterns(df_japan)
        
        # 2. マグニチュードトレンド
        magnitude_trends = explore_magnitude_trends(df_japan)
        
        # 3. 地震発生間隔
        intervals, interval_stats = explore_time_intervals(df_japan)
        
        # 4. 世界地震と国内地震の相関
        world_japan_merged, correlations = explore_world_japan_correlation(df_world, df_japan)
        
        # 5. 深さとマグニチュードの関係
        depth_mag_merged, depth_correlations = explore_depth_magnitude_relationship(df_depth, df_japan)
        
        # 6. 季節性パターン
        monthly_stats, seasonal_stats = explore_seasonal_patterns(df_japan)
        
        # 7. マグニチュード分布の特徴量
        dist_features = explore_magnitude_distribution_features(df_japan)
        
        # サマリーレポート作成
        create_feature_summary_report()
        
    except Exception as e:
        print(f"✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("全ての特徴探索が完了しました！")
    print("=" * 60)
    print(f"\n出力ファイルは {OUTPUT_DIR} に保存されました。")


if __name__ == "__main__":
    main()
