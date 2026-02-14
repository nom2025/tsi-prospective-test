"""
世界地震データ自動取得スクリプト
USGS Earthquake Catalog APIから世界地震データを取得し、world-all.csvに追加します。
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import sys

# 設定
BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
DATA_DIR = Path(__file__).parent
CSV_FILE = DATA_DIR.parent / "data" / "world-all.csv"
# プロジェクトルートの生データフォルダへのパス
PROJECT_ROOT = DATA_DIR.parent.parent
RAW_DATA_FILE = PROJECT_ROOT / "生データ" / "world-all.csv"
MIN_MAGNITUDE = 6.0  # 最小マグニチュード

# 州名・地域名を国名にマッピングする辞書
STATE_TO_COUNTRY = {
    # アメリカ合衆国の州
    'Alaska': 'United States',
    'Oregon': 'United States',
    'California': 'United States',
    'Hawaii': 'United States',
    'Washington': 'United States',
    'Nevada': 'United States',
    'Idaho': 'United States',
    'Montana': 'United States',
    'Wyoming': 'United States',
    'Utah': 'United States',
    'Arizona': 'United States',
    'New Mexico': 'United States',
    'Texas': 'United States',
    'Oklahoma': 'United States',
    'Kansas': 'United States',
    'Nebraska': 'United States',
    'South Dakota': 'United States',
    'North Dakota': 'United States',
    'Minnesota': 'United States',
    'Iowa': 'United States',
    'Missouri': 'United States',
    'Arkansas': 'United States',
    'Louisiana': 'United States',
    'Mississippi': 'United States',
    'Alabama': 'United States',
    'Tennessee': 'United States',
    'Kentucky': 'United States',
    'Illinois': 'United States',
    'Indiana': 'United States',
    'Ohio': 'United States',
    'Michigan': 'United States',
    'Wisconsin': 'United States',
    'Pennsylvania': 'United States',
    'New York': 'United States',
    'Vermont': 'United States',
    'New Hampshire': 'United States',
    'Maine': 'United States',
    'Massachusetts': 'United States',
    'Rhode Island': 'United States',
    'Connecticut': 'United States',
    'New Jersey': 'United States',
    'Delaware': 'United States',
    'Maryland': 'United States',
    'Virginia': 'United States',
    'West Virginia': 'United States',
    'North Carolina': 'United States',
    'South Carolina': 'United States',
    'Georgia': 'United States',
    'Florida': 'United States',
    # その他の地域名
    'Puerto Rico': 'United States',
    'Guam': 'United States',
    'American Samoa': 'United States',
    'Northern Mariana Islands': 'United States',
    'U.S. Virgin Islands': 'United States',
}


def extract_country_name(place):
    """
    USGS APIのplaceフィールドから国名を抽出
    
    Args:
        place: USGS APIのplaceフィールド（例: "295 km W of Bandon, Oregon"）
    
    Returns:
        str: 国名（例: "United States"）
    """
    if not place or place == 'nan':
        return place
    
    place = place.strip()
    
    # カンマで分割して、最後の部分を取得
    if ',' in place:
        parts = place.split(',')
        last_part = parts[-1].strip()
        
        # 州名や地域名を国名にマッピング
        if last_part in STATE_TO_COUNTRY:
            return STATE_TO_COUNTRY[last_part]
        
        # マッピングがない場合は、そのまま返す（国名の可能性が高い）
        return last_part
    else:
        # カンマがない場合は、そのまま返す（国名の可能性が高い）
        # ただし、州名の可能性もあるので、マッピングを確認
        if place in STATE_TO_COUNTRY:
            return STATE_TO_COUNTRY[place]
        return place


def get_latest_date_from_csv():
    """
    world-all.csvから最新の日付を取得
    
    Returns:
        datetime: 最新の日付（存在しない場合はNone）
    """
    if not CSV_FILE.exists():
        return None
    
    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            return None
        
        # 日付をパース
        df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d', errors='coerce')
        df = df.dropna(subset=['date'])
        
        if df.empty:
            return None
        
        # 最新の日付を取得
        latest_date = df['date'].max()
        return latest_date
    except Exception as e:
        print(f"既存データの読み込みエラー: {e}")
        return None


def fetch_earthquake_data(start_date, end_date=None, debug_mode=False):
    """
    USGS APIから地震データを取得
    
    Args:
        start_date: 開始日時（datetime）
        end_date: 終了日時（datetime、Noneの場合は現在日時）
        debug_mode: デバッグモード（Trueの場合、詳細なログを出力）
    
    Returns:
        list: 地震データのリスト [{'date': '2025/12/20', 'place_w': 'Japan', 'magnitude_w': 6.5}, ...]
    """
    if end_date is None:
        end_date = datetime.now()
    
    # APIパラメータ
    params = {
        'format': 'csv',
        'starttime': start_date.strftime('%Y-%m-%d'),
        'endtime': end_date.strftime('%Y-%m-%d'),
        'minmagnitude': MIN_MAGNITUDE,
        'orderby': 'time'  # 時間順
    }
    
    print(f"データ取得を開始します...")
    print(f"URL: {BASE_URL}")
    print(f"パラメータ:")
    print(f"  開始日時: {start_date.strftime('%Y-%m-%d')}")
    print(f"  終了日時: {end_date.strftime('%Y-%m-%d')}")
    print(f"  最小マグニチュード: {MIN_MAGNITUDE}")
    
    all_data = []
    
    try:
        # APIリクエスト（最大20,000件まで取得可能）
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        # レスポンスの内容を確認（デバッグ用）
        if debug_mode:
            print(f"APIレスポンスステータス: {response.status_code}")
            print(f"レスポンスの最初の500文字: {response.text[:500]}")
        
        # CSVデータをパース
        from io import StringIO
        
        # 空のレスポンスをチェック
        response_text = response.text.strip()
        if not response_text:
            print("APIから空のレスポンスが返されました")
            return []
        
        # CSVヘッダーを確認
        lines = response_text.split('\n')
        if len(lines) <= 1:
            print("データが見つかりませんでした（CSVにデータ行がありません）")
            if debug_mode:
                print(f"レスポンス内容: {response_text[:500]}")
            return []
        
        # CSVをパース
        csv_data = StringIO(response_text)
        try:
            df = pd.read_csv(csv_data)
        except Exception as e:
            print(f"CSVパースエラー: {e}")
            if debug_mode:
                print(f"レスポンス内容（最初の1000文字）: {response_text[:1000]}")
            return []
        
        if df.empty:
            print("データが見つかりませんでした（DataFrameが空です）")
            if debug_mode:
                print(f"レスポンス内容（最初の1000文字）: {response_text[:1000]}")
            return []
        
        print(f"取得したデータ: {len(df)} 件")
        
        # 必要なカラムを抽出・変換
        if debug_mode:
            print(f"取得したCSVのカラム: {list(df.columns)}")
        
        for _, row in df.iterrows():
            try:
                # 日時から日付のみを抽出
                # USGS APIのCSV形式では 'time' カラムにISO形式の日時が含まれる
                time_str = str(row.get('time', ''))
                if not time_str or time_str == 'nan':
                    continue
                
                # ISO形式の日時から日付部分を抽出
                date_obj = pd.to_datetime(time_str)
                date_str = date_obj.strftime('%Y/%m/%d')
                
                # 震源地を取得（placeカラム）
                place_raw = str(row.get('place', '')).strip()
                if not place_raw or place_raw == 'nan':
                    # placeがない場合は緯度・経度から推測
                    lat = row.get('latitude', None)
                    lon = row.get('longitude', None)
                    if pd.notna(lat) and pd.notna(lon):
                        place = f"Lat:{lat:.2f}, Lon:{lon:.2f}"
                    else:
                        continue
                else:
                    # USGS APIのplaceフィールドから国名を抽出
                    place = extract_country_name(place_raw)
                
                # マグニチュードを取得
                magnitude = row.get('mag', None)
                if pd.isna(magnitude):
                    continue
                
                magnitude = float(magnitude)
                if magnitude < MIN_MAGNITUDE:
                    continue
                
                all_data.append({
                    'date': date_str,
                    'place_w': place,
                    'magnitude_w': magnitude
                })
            except Exception as e:
                # 個別のデータ処理エラーはスキップ
                if debug_mode:
                    print(f"データ処理エラー: {e}, 行: {row}")
                continue
        
        print(f"処理済みデータ: {len(all_data)} 件")
        
    except requests.RequestException as e:
        print(f"✗ APIリクエストエラー: {e}")
        return []
    except Exception as e:
        print(f"✗ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    return all_data


def load_existing_data():
    """既存のCSVファイルを読み込む"""
    if not CSV_FILE.exists():
        print(f"既存のファイルが見つかりません: {CSV_FILE}")
        return pd.DataFrame(columns=['date', 'place_w', 'magnitude_w'])
    
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"既存データ: {len(df)} 件")
        return df
    except Exception as e:
        print(f"既存データの読み込みエラー: {e}")
        return pd.DataFrame(columns=['date', 'place_w', 'magnitude_w'])


def find_new_data(existing_df, new_data):
    """
    新しいデータを特定
    
    Args:
        existing_df: 既存のDataFrame
        new_data: 新しく取得したデータのリスト
    
    Returns:
        DataFrame: 新しいデータのみを含むDataFrame
    """
    if existing_df.empty:
        return pd.DataFrame(new_data)
    
    # 既存データのセットを作成（重複チェック用）
    existing_set = set()
    for _, row in existing_df.iterrows():
        # 日付を正規化（YYYY/MM/DD形式に統一）
        date_str = str(row['date']).strip()
        place = str(row['place_w']).strip()
        magnitude = float(row['magnitude_w'])
        key = (date_str, place, magnitude)
        existing_set.add(key)
    
    # 新しいデータをフィルタリング
    new_records = []
    for record in new_data:
        key = (record['date'], record['place_w'], record['magnitude_w'])
        if key not in existing_set:
            new_records.append(record)
    
    return pd.DataFrame(new_records)


def create_work_folder():
    """作業日の日付でフォルダを作成"""
    today = datetime.now().strftime('%Y%m%d')
    work_folder = DATA_DIR / today
    
    if not work_folder.exists():
        work_folder.mkdir(parents=True, exist_ok=True)
        print(f"作業フォルダを作成しました: {work_folder}")
    else:
        print(f"作業フォルダは既に存在します: {work_folder}")
    
    return work_folder


def save_backup(df, work_folder):
    """バックアップを保存"""
    backup_file = work_folder / f"world_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(backup_file, index=False, encoding='utf-8-sig')
    print(f"バックアップを保存しました: {backup_file}")


def main():
    """メイン処理"""
    import sys
    
    # デバッグモードのチェック
    debug_mode = '--debug' in sys.argv or '-d' in sys.argv
    
    print("=" * 60)
    print("世界地震データ自動取得スクリプト")
    print("=" * 60)
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"データファイル: {CSV_FILE}")
    if debug_mode:
        print("デバッグモード: ON")
    print("-" * 60)
    
    # 1. 作業フォルダを作成
    work_folder = create_work_folder()
    
    # 2. 既存データを読み込む
    print("\n[既存データの読み込み]")
    existing_df = load_existing_data()
    
    # 3. 開始日時を決定
    print("\n[開始日時の決定]")
    
    # 既存データの最新日付を取得して表示
    if not existing_df.empty:
        existing_df['date_parsed'] = pd.to_datetime(existing_df['date'], format='%Y/%m/%d', errors='coerce')
        latest_date = existing_df['date_parsed'].max()
        if pd.notna(latest_date):
            print(f"既存データの最新日時: {latest_date.strftime('%Y-%m-%d')}")
            # 同じ日から開始（重複チェックで除外される）
            start_date = latest_date
        else:
            start_date = datetime(2010, 1, 1)
            print(f"既存データの日付が無効です。デフォルト開始日時を使用: {start_date.strftime('%Y-%m-%d')}")
    else:
        # 既存データがない場合、デフォルトの開始日時を設定
        start_date = datetime(2010, 1, 1)
        print(f"既存データが見つかりません。デフォルト開始日時を使用: {start_date.strftime('%Y-%m-%d')}")
    
    print(f"開始日時: {start_date.strftime('%Y-%m-%d')}")
    
    # デバッグモードでは直近7日間のみ取得
    if debug_mode:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        print(f"デバッグモード: 直近7日間のデータを取得します")
    
    # 4. 新しいデータを取得
    print("\n[データ取得]")
    # 終了日時を現在日時に設定
    end_date = datetime.now()
    new_data = fetch_earthquake_data(start_date, end_date=end_date, debug_mode=debug_mode)
    
    if not new_data:
        print("新しいデータが取得できませんでした")
        if debug_mode:
            print("\n[デバッグ情報]")
            print(f"開始日時: {start_date.strftime('%Y-%m-%d')}")
            print(f"終了日時: {datetime.now().strftime('%Y-%m-%d')}")
        return
    
    # 5. 新しいデータを特定
    print("\n[新しいデータの特定]")
    new_df = find_new_data(existing_df, new_data)
    
    if new_df.empty:
        print("新しいデータはありませんでした")
        if debug_mode:
            print("\n[デバッグ情報]")
            print(f"取得したデータ数: {len(new_data)}")
            if new_data:
                print("取得したデータのサンプル（最初の5件）:")
                for i, record in enumerate(new_data[:5], 1):
                    print(f"  {i}. {record}")
        
        # 新しいデータがなくても、既存データを生データフォルダにコピー
        print("\n[生データフォルダへのコピー]")
        try:
            RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            existing_df.to_csv(RAW_DATA_FILE, index=False, encoding='utf-8-sig')
            print(f"✓ 生データフォルダにコピーしました: {RAW_DATA_FILE}")
        except Exception as e:
            print(f"⚠ 生データフォルダへのコピーに失敗しました: {e}")
        
        print("\n" + "=" * 60)
        print("処理が完了しました")
        print("=" * 60)
        return
    
    print(f"新しいデータ: {len(new_df)} 件")
    
    # 6. バックアップを保存
    print("\n[バックアップ作成]")
    save_backup(existing_df, work_folder)
    
    # 7. データを追加
    print("\n[データ追加]")
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # 日付でソート（新しいデータが下に来るように）
    updated_df['date_parsed'] = pd.to_datetime(updated_df['date'], format='%Y/%m/%d', errors='coerce')
    updated_df = updated_df.sort_values('date_parsed', ascending=False)
    updated_df = updated_df.drop('date_parsed', axis=1)
    
    # CSVに保存（データ自動取得フォルダ）
    updated_df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
    print(f"✓ データを追加しました: {CSV_FILE}")
    print(f"  既存: {len(existing_df)} 件")
    print(f"  追加: {len(new_df)} 件")
    print(f"  合計: {len(updated_df)} 件")
    
    # 7-1. 生データフォルダにもコピー
    print("\n[生データフォルダへのコピー]")
    try:
        # 生データフォルダが存在しない場合は作成
        RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        updated_df.to_csv(RAW_DATA_FILE, index=False, encoding='utf-8-sig')
        print(f"✓ 生データフォルダにコピーしました: {RAW_DATA_FILE}")
    except Exception as e:
        print(f"⚠ 生データフォルダへのコピーに失敗しました: {e}")
        print(f"  データは {CSV_FILE} に保存されています")
    
    # 8. 新しいデータをログファイルに保存
    log_file = work_folder / f"world_new_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    new_df.to_csv(log_file, index=False, encoding='utf-8-sig')
    print(f"✓ 新しいデータのログを保存しました: {log_file}")
    
    print("\n" + "=" * 60)
    print("処理が完了しました")
    print("=" * 60)


if __name__ == "__main__":
    main()
