"""
地震データ自動取得スクリプト
Yahoo!天気・災害から地震データを取得し、2010-all.csvに追加します。
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
from pathlib import Path
import time
import os

# 設定
BASE_URL = "https://typhoon.yahoo.co.jp/weather/jp/earthquake/list/"
DATA_DIR = Path(__file__).parent
CSV_FILE = DATA_DIR.parent / "data" / "2010-all.csv"
# プロジェクトルートの生データフォルダへのパス
PROJECT_ROOT = DATA_DIR.parent.parent
RAW_DATA_FILE = PROJECT_ROOT / "生データ" / "2010-all.csv"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def get_current_year():
    """現在の年を取得"""
    return datetime.now().year


def fetch_earthquake_data(max_pages=10):
    """
    Yahoo!天気・災害から地震データを取得
    
    Args:
        max_pages: 取得する最大ページ数（デフォルト: 10）
    
    Returns:
        list: 地震データのリスト [{'date': '12月20日', 'place': '震源地', 'magnitude': 4.5}, ...]
    """
    all_data = []
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})
    
    print(f"データ取得を開始します...")
    print(f"URL: {BASE_URL}")
    
    for page in range(1, max_pages + 1):
        try:
            # ページネーション対応（Yahoo!のページネーション形式を確認）
            if page == 1:
                url = BASE_URL
            else:
                # 次の100件へのリンクを探す
                url = f"{BASE_URL}?sort=0&key={page}"
            
            print(f"  ページ {page} を取得中...", end=" ", flush=True)
            response = session.get(url, timeout=15)
            response.encoding = 'utf-8'
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # テーブルを探す（複数のテーブルがある可能性があるので、適切なものを選択）
            tables = soup.find_all('table')
            table = None
            for t in tables:
                # ヘッダーに「発生時刻」「震源地」「マグニチュード」が含まれるテーブルを探す
                headers = t.find_all('th')
                if headers and any('発生時刻' in h.get_text() or '震源地' in h.get_text() for h in headers):
                    table = t
                    break
            
            if not table:
                # テーブルが見つからない場合、最初のテーブルを使用
                if tables:
                    table = tables[0]
                else:
                    print("テーブルが見つかりません")
                    break
            
            # テーブルの行を取得（ヘッダー行を除く）
            rows = table.find_all('tr')
            if len(rows) <= 1:
                print("データが見つかりません")
                break
            
            # 最初の行がヘッダーの可能性があるので、2行目から開始
            data_rows = rows[1:] if len(rows) > 1 else rows
            
            page_data_count = 0
            for row in data_rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 3:
                    # 日付、震源地、マグニチュードを取得
                    # 日付はリンク内にある可能性がある
                    date_cell = cells[0]
                    date_link = date_cell.find('a')
                    if date_link:
                        date_text = date_link.get_text(strip=True)
                    else:
                        date_text = date_cell.get_text(strip=True)
                    
                    place = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                    magnitude_text = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                    
                    # 空のデータをスキップ
                    if not date_text or not place or not magnitude_text:
                        continue
                    
                    # 日付を変換（例: "2025年12月20日 15時45分ごろ" -> "2025/12/20"）
                    date_formatted = convert_date_format(date_text)
                    
                    # マグニチュードを数値に変換
                    try:
                        magnitude = float(magnitude_text)
                        all_data.append({
                            'date': date_formatted,
                            'place': place,
                            'magnitude': magnitude
                        })
                        page_data_count += 1
                    except ValueError:
                        continue
            
            print(f"✓ {page_data_count}件のデータを取得")
            
            # データが取得できなかった場合は終了
            if page_data_count == 0:
                print("これ以上データがありません")
                break
            
            # レート制限を避けるため少し待機
            time.sleep(1)
            
        except requests.RequestException as e:
            print(f"✗ エラー: {e}")
            break
        except Exception as e:
            print(f"✗ 予期しないエラー: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print(f"\n合計 {len(all_data)} 件のデータを取得しました")
    return all_data


def convert_date_format(date_text):
    """
    日付文字列を変換
    例: "2025年12月20日 15時45分ごろ" -> "2025/12/20"
    例: "12月20日" -> "2025/12/20" (現在の年を使用)
    """
    # 年、月、日を抽出（完全な形式: "2025年12月20日"）
    match = re.search(r'(\d+)年(\d+)月(\d+)日', date_text)
    if match:
        year = match.group(1)
        month = match.group(2).zfill(2)  # 2桁にゼロ埋め
        day = match.group(3).zfill(2)    # 2桁にゼロ埋め
        return f"{year}/{month}/{day}"
    
    # 既に "12月20日" 形式の場合は現在の年を使用
    match_month_day = re.match(r'(\d+)月(\d+)日', date_text)
    if match_month_day:
        current_year = get_current_year()
        month = match_month_day.group(1).zfill(2)  # 2桁にゼロ埋め
        day = match_month_day.group(2).zfill(2)    # 2桁にゼロ埋め
        return f"{current_year}/{month}/{day}"
    
    # 既に "YYYY/MM/DD" 形式の場合はそのまま返す
    if re.match(r'\d{4}/\d{2}/\d{2}', date_text):
        return date_text
    
    return date_text


def normalize_date_in_dataframe(df):
    """
    DataFrame内の日付をyyyy/mm/dd形式に統一する
    
    Args:
        df: 日付カラムを含むDataFrame
    
    Returns:
        DataFrame: 日付が統一されたDataFrame
    """
    if df.empty:
        return df
    
    # 日付カラムをyyyy/mm/dd形式に変換
    df['date'] = df['date'].astype(str).apply(convert_date_format)
    return df


def load_existing_data():
    """既存のCSVファイルを読み込む（日付形式を統一）"""
    if not CSV_FILE.exists():
        print(f"既存のファイルが見つかりません: {CSV_FILE}")
        return pd.DataFrame(columns=['date', 'place', 'magnitude'])
    
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"既存データ: {len(df)} 件")
        # 日付形式を統一
        df = normalize_date_in_dataframe(df)
        return df
    except Exception as e:
        print(f"既存データの読み込みエラー: {e}")
        return pd.DataFrame(columns=['date', 'place', 'magnitude'])


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
        key = (str(row['date']), str(row['place']), float(row['magnitude']))
        existing_set.add(key)
    
    # 新しいデータをフィルタリング
    new_records = []
    for record in new_data:
        key = (record['date'], record['place'], record['magnitude'])
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
    backup_file = work_folder / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(backup_file, index=False, encoding='utf-8-sig')
    print(f"バックアップを保存しました: {backup_file}")


def main():
    """メイン処理"""
    import sys
    
    # デバッグモードのチェック
    debug_mode = '--debug' in sys.argv or '-d' in sys.argv
    
    print("=" * 60)
    print("地震データ自動取得スクリプト")
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
    
    # 3. 新しいデータを取得
    print("\n[データ取得]")
    # デバッグモードでは1ページのみ取得
    max_pages = 1 if debug_mode else 10
    new_data = fetch_earthquake_data(max_pages=max_pages)
    
    if not new_data:
        print("新しいデータが取得できませんでした")
        return
    
    # 4. 新しいデータを特定
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
        
        # 新しいデータがなくても、既存データの日付形式を統一して保存
        print("\n[既存データの日付形式を統一]")
        normalized_df = normalize_date_in_dataframe(existing_df.copy())
        
        # 日付でソート
        normalized_df['date_parsed'] = pd.to_datetime(normalized_df['date'], format='%Y/%m/%d', errors='coerce')
        normalized_df = normalized_df.sort_values('date_parsed', ascending=False)
        normalized_df = normalized_df.drop('date_parsed', axis=1)
        
        # CSVに保存（データ自動取得フォルダ）
        normalized_df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
        print(f"✓ 日付形式を統一して保存しました: {CSV_FILE}")
        
        # 生データフォルダにもコピー
        print("\n[生データフォルダへのコピー]")
        try:
            RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            normalized_df.to_csv(RAW_DATA_FILE, index=False, encoding='utf-8-sig')
            print(f"✓ 生データフォルダにコピーしました: {RAW_DATA_FILE}")
        except Exception as e:
            print(f"⚠ 生データフォルダへのコピーに失敗しました: {e}")
        
        print("\n" + "=" * 60)
        print("処理が完了しました")
        print("=" * 60)
        return
    
    print(f"新しいデータ: {len(new_df)} 件")
    
    # 5. バックアップを保存
    print("\n[バックアップ作成]")
    save_backup(existing_df, work_folder)
    
    # 6. データを追加
    print("\n[データ追加]")
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # 日付形式を統一（念のため）
    updated_df = normalize_date_in_dataframe(updated_df)
    
    # 日付でソート（新しいデータが上に来るように）
    updated_df['date_parsed'] = pd.to_datetime(updated_df['date'], format='%Y/%m/%d', errors='coerce')
    updated_df = updated_df.sort_values('date_parsed', ascending=False)
    updated_df = updated_df.drop('date_parsed', axis=1)
    
    # CSVに保存（データ自動取得フォルダ）
    updated_df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
    print(f"✓ データを追加しました: {CSV_FILE}")
    print(f"  既存: {len(existing_df)} 件")
    print(f"  追加: {len(new_df)} 件")
    print(f"  合計: {len(updated_df)} 件")
    
    # 7. 生データフォルダにもコピー
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
    log_file = work_folder / f"new_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    new_df.to_csv(log_file, index=False, encoding='utf-8-sig')
    print(f"✓ 新しいデータのログを保存しました: {log_file}")
    
    print("\n" + "=" * 60)
    print("処理が完了しました")
    print("=" * 60)


if __name__ == "__main__":
    main()
