"""
GPS・地殻変動データ自動取得スクリプト
国土地理院「GEONET」からGPS・地殻変動データを取得し、gnss_data.csvに追加します。
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime, timedelta
from pathlib import Path
import time
import sys
import os
import ftplib

# 設定
GEONET_BASE_URL = "https://terras.gsi.go.jp/"
GEONET_POS_URL = "https://terras.gsi.go.jp/pos_main.php"
GEONET_FTP_HOST = "terras.gsi.go.jp"
GEONET_FTP_PATH = "/pub/GPS/"
DATA_DIR = Path(__file__).parent
CSV_FILE = DATA_DIR.parent / "data" / "gnss_data.csv"
# データディレクトリが存在しない場合は作成
CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
# プロジェクトルートの生データフォルダへのパス
PROJECT_ROOT = DATA_DIR.parent.parent
RAW_DATA_FILE = PROJECT_ROOT / "生データ" / "gnss_data.csv"
START_DATE = datetime(2010, 1, 1)  # データ取得開始日
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
# 解析種別: F5解（現在の解析ストラテジ）
ANALYSIS_TYPE = "F5"

# 主要な観測地点（GEONETの電子基準点コード）
# データ取得可能な7地点のみを対象とする
STATIONS = {
    '950421': {'name': '東京', 'lat': 35.6895, 'lon': 139.6917},
    '950447': {'name': '横浜', 'lat': 35.4437, 'lon': 139.6380},
    '950456': {'name': '静岡', 'lat': 34.9756, 'lon': 138.3826},
    '950465': {'name': '名古屋', 'lat': 35.1815, 'lon': 136.9066},
    '950474': {'name': '大阪', 'lat': 34.6937, 'lon': 135.5023},
    '950483': {'name': '神戸', 'lat': 34.6901, 'lon': 135.1956},
    '950492': {'name': '広島', 'lat': 34.3853, 'lon': 132.4553},
}


def parse_pos_file(content, station_id, station_info, year, debug_mode=False):
    """
    POSファイル（日々の座標値ファイル）をパース
    
    Args:
        content: POSファイルの内容（テキスト）
        station_id: 電子基準点コード
        station_info: 観測地点情報
        year: 年
        debug_mode: デバッグモード
    
    Returns:
        list: GPSデータのリスト
    """
    all_data = []
    
    lines = content.split('\n')
    
    # デバッグ用: POSファイルの最初の30行を表示（データ行を含む）
    if debug_mode:
        print(f"\n    POSファイルの最初の30行:")
        for i, line in enumerate(lines[:30], 1):
            print(f"      {i}: {line[:150]}")
    
    # POSファイルの構造を確認
    # ヘッダー: +DATA行の後にデータが始まる
    # データ行の形式: yyyy mm dd HH:MM:SS X (m) Y (m) Z (m) Lat. (deg.) Lon. (deg.) Height (m)
    # 例: 2025 01 01 12:00:00 -3.6869605979E+06  3.8102068721E+06  3.5359610890E+06  3.3877516012E+01  1.3405819693E+02  1.4100849686E+03
    # 数値は科学記数法（E+06形式）で表現される
    
    data_started = False
    header_found = False
    reference_x = None
    reference_y = None
    reference_z = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # +DATA行を検出
        if line.startswith('+DATA'):
            data_started = True
            header_found = False
            continue
        
        # -DATA行でデータ終了
        if line.startswith('-DATA'):
            data_started = False
            continue
        
        # ヘッダー行をスキップ（*で始まる行）
        if line.startswith('*'):
            if 'yyyy' in line.lower() or 'mm' in line.lower() or 'dd' in line.lower():
                header_found = True
            continue
        
        # データ行をパース（+DATA行の後で、*で始まらない行）
        if data_started and header_found and not line.startswith('*') and not line.startswith('+') and not line.startswith('-'):
            # 空白で分割
            parts = re.split(r'\s+', line.strip())
            if len(parts) < 6:  # 最低限、日付と時刻、X, Y, Z座標が必要
                continue
            
            try:
                # 日付と時刻を抽出: yyyy mm dd HH:MM:SS
                if len(parts) >= 5:
                    year = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    date_str = f"{year}/{month:02d}/{day:02d}"
                    
                    # X, Y, Z座標を抽出（科学記数法に対応）
                    # 形式: yyyy mm dd HH:MM:SS X Y Z Lat Lon Height
                    # parts[3] = HH:MM:SS, parts[4] = X, parts[5] = Y, parts[6] = Z
                    x_m = None
                    y_m = None
                    z_m = None
                    
                    # 時刻の後の数値を抽出（科学記数法に対応）
                    if len(parts) >= 7:
                        try:
                            # X座標（parts[4]）- 負の値
                            x_m = float(parts[4])
                            # Y座標（parts[5]）- 正の値
                            y_m = float(parts[5])
                            # Z座標（parts[6]）- 正の値
                            z_m = float(parts[6])
                        except (ValueError, IndexError) as e:
                            if debug_mode:
                                print(f"\n    座標値抽出エラー: {e}, 行: {line[:100]}")
                            continue
                    
                    # 座標値が取得できた場合のみデータを追加
                    if x_m is not None and y_m is not None and z_m is not None:
                        # 最初の日の座標を基準として設定
                        if reference_x is None:
                            reference_x = x_m
                            reference_y = y_m
                            reference_z = z_m
                        
                        # 基準座標からの変動量を計算（メートル単位）
                        # 注意: X, Y, Zは地心直交座標（ECEF座標）なので、
                        # 直接東西・南北・上下の変動量ではないが、変動量の大きさを分析可能
                        delta_x = x_m - reference_x
                        delta_y = y_m - reference_y
                        delta_z = z_m - reference_z
                        
                        # メートルからミリメートルに変換して保存
                        all_data.append({
                            'date': date_str,
                            'station_id': station_id,
                            'station_name': station_info['name'],
                            'latitude': station_info['lat'],
                            'longitude': station_info['lon'],
                            'east_mm': delta_x * 1000.0,  # 基準からの変動量（mm）
                            'north_mm': delta_y * 1000.0,
                            'up_mm': delta_z * 1000.0
                        })
            except (ValueError, IndexError) as e:
                if debug_mode:
                    print(f"\n    データ行パースエラー: {e}, 行: {line[:100]}")
                continue
    
    return all_data


def fetch_gnss_data_for_year(station_id, station_info, year, debug_mode=False):
    """
    指定された観測地点・年のGPS・地殻変動データを取得
    
    Args:
        station_id: 電子基準点コード
        station_info: 観測地点情報（name, lat, lon）
        year: 年
        debug_mode: デバッグモード
    
    Returns:
        list: GPSデータのリスト [{'date': '2010/01/01', 'station_id': '950421', ...}, ...]
    """
    all_data = []
    
    print(f"  {station_info['name']}({station_id}) - {year}年のデータを取得中...", end=" ", flush=True)
    
    # ファイル名の規則: nnnnn[n].yy.pos
    # nnnnn[n]: 電子基準点の番号（5または6桁）
    # yy: 観測した日の西暦下2桁
    year_last2 = str(year)[-2:]
    filename = f"{station_id}.{year_last2}.pos"
    
    # FTP経由でデータを取得（試行）
    # CI環境ではFTPポートがブロック/遅延されるためHTTP直行
    file_content = None
    ftp_success = False
    skip_ftp = (os.environ.get('CI') == 'true')

    if skip_ftp:
        print("(FTPスキップ→HTTP)", end=" ", flush=True)

    ftp = None
    try:
        if skip_ftp:
            raise Exception("CI環境のためFTPスキップ")
        ftp = ftplib.FTP(GEONET_FTP_HOST, timeout=30)
        # 匿名ログイン（ユーザー名: anonymous, パスワード: 空またはメールアドレス）
        try:
            ftp.login('anonymous', '')
            ftp_success = True
        except ftplib.error_perm:
            # パスワードなしで試す
            try:
                ftp.login('anonymous')
                ftp_success = True
            except ftplib.error_perm:
                if debug_mode:
                    print(f"\n    FTP匿名ログイン失敗。Web経由で取得を試みます...")
                ftp_success = False
        
        if ftp_success:
            # FTP接続が成功した場合のみ処理を続行
            # 年ごとのディレクトリ構造を確認
            # 一般的な構造: /pub/GPS/F5/YYYY/ または /pub/GPS/pos/YYYY/
            ftp_paths = [
                f"/pub/GPS/{ANALYSIS_TYPE}/{year}/",
                f"/pub/GPS/pos/{year}/",
                f"/pub/GPS/{year}/",
                f"/pub/GPS/F5/{year}/",
            ]
            
            for ftp_path in ftp_paths:
                try:
                    ftp.cwd(ftp_path)
                    if debug_mode:
                        print(f"\n    FTPパス: {ftp_path}")
                    
                    # ファイル一覧を取得
                    files = ftp.nlst()
                    if filename in files:
                        # ファイルをダウンロード
                        from io import BytesIO
                        bio = BytesIO()
                        ftp.retrbinary(f'RETR {filename}', bio.write)
                        file_content = bio.getvalue().decode('utf-8', errors='ignore')
                        ftp_success = True
                        if debug_mode:
                            print(f"    ✓ FTP経由でファイルを取得: {filename}")
                        break
                except ftplib.error_perm:
                    continue
        
        if ftp:
            try:
                ftp.quit()
            except:
                pass
                
    except Exception as e:
        if debug_mode:
            print(f"\n    FTPエラー: {e}")
        ftp_success = False
    
    # FTPで取得できない場合は、Web経由で取得を試みる
    if not ftp_success or file_content is None:
        if debug_mode:
            print(f"\n    FTPで取得できませんでした。Web経由で取得を試みます...")
        
        try:
            # Webページからダウンロードリンクを取得
            session = requests.Session()
            session.headers.update({'User-Agent': USER_AGENT})
            
            # 直接ファイルをダウンロードするURLを構築
            # 例: https://terras.gsi.go.jp/pub/GPS/F5/2025/950421.25.pos
            download_urls = [
                f"{GEONET_BASE_URL}pub/GPS/{ANALYSIS_TYPE}/{year}/{filename}",
                f"{GEONET_BASE_URL}pub/GPS/pos/{year}/{filename}",
                f"{GEONET_BASE_URL}pub/GPS/{year}/{filename}",
            ]
            
            for download_url in download_urls:
                try:
                    if debug_mode:
                        print(f"    ダウンロードURLを試行: {download_url}")
                    file_response = session.get(download_url, timeout=30)
                    if debug_mode:
                        print(f"      ステータスコード: {file_response.status_code}")
                    if file_response.status_code == 200:
                        file_content = file_response.text
                        if file_content and len(file_content) > 100:  # ファイルが空でないことを確認
                            if debug_mode:
                                print(f"    ✓ Web経由でファイルを取得: {filename} ({len(file_content)}文字)")
                            break
                        elif debug_mode:
                            print(f"      ファイルが空または無効です")
                except requests.RequestException as e:
                    if debug_mode:
                        print(f"      リクエストエラー: {e}")
                    continue
            
            # ダウンロードURLが直接取得できない場合は、POSTリクエストでダウンロード
            if file_content is None:
                if debug_mode:
                    print(f"    POSTリクエストでダウンロードを試みます...")
                
                try:
                    # pos_download.phpにPOSTリクエストを送信
                    download_url = f"{GEONET_BASE_URL}pos_download.php"
                    post_data = {
                        'sol': ANALYSIS_TYPE,  # 解析種別（F5）
                        'observation_cd': station_id,  # 電子基準点コード
                        'YEAR': year  # 年
                    }
                    
                    if debug_mode:
                        print(f"    ダウンロードURL: {download_url}")
                        print(f"    POSTデータ: {post_data}")
                    
                    # POSTリクエストを送信
                    file_response = session.post(download_url, data=post_data, timeout=30)
                    
                    if debug_mode:
                        print(f"    レスポンスステータス: {file_response.status_code}")
                        print(f"    コンテンツタイプ: {file_response.headers.get('content-type', 'unknown')}")
                    
                    # レスポンスが成功で、ファイル内容が含まれているか確認
                    if file_response.status_code == 200:
                        file_content = file_response.text
                        
                        # デバッグ用: レスポンスの最初の部分を確認
                        if debug_mode:
                            print(f"    レスポンスの最初の500文字: {file_content[:500]}")
                            # レスポンス全体をデバッグファイルに保存
                            debug_file = DATA_DIR.parent / "debug" / f"post_response_{station_id}_{year}.html"
                            debug_file.parent.mkdir(parents=True, exist_ok=True)
                            with open(debug_file, 'w', encoding='utf-8') as f:
                                f.write(file_content)
                            print(f"    レスポンスを保存しました: {debug_file}")
                        
                        # HTMLページが返された場合は、POSファイルへのリンクを探す
                        if file_content.startswith('<!') or '<html' in file_content.lower() or '<body' in file_content.lower():
                            if debug_mode:
                                print(f"    HTMLページが返されました。POSファイルへのリンクを探します...")
                            
                            # HTMLをパースしてPOSファイルへのリンクを探す
                            soup = BeautifulSoup(file_content, 'html.parser')
                            
                            # リンクを探す（例: <a href='unt/file_download.php'>950421.25.pos</a>）
                            links = soup.find_all('a', href=True)
                            download_link = None
                            
                            if debug_mode:
                                print(f"    検出されたリンク数: {len(links)}")
                            
                            for link in links:
                                href = link.get('href', '')
                                link_text = link.get_text(strip=True)
                                
                                if debug_mode:
                                    print(f"      リンク候補: text='{link_text}', href='{href}'")
                                
                                # POSファイル名が含まれるリンクを探す
                                # 大文字小文字を区別しない検索
                                if (filename.lower() in link_text.lower() or 
                                    '.pos' in href.lower() or 
                                    'file_download' in href.lower()):
                                    download_link = href
                                    if debug_mode:
                                        print(f"    ✓ リンクを発見: {link_text}, href: {href}")
                                    break
                            
                            # リンクが見つからない場合は、テーブル内のセルから直接探す
                            if not download_link:
                                if debug_mode:
                                    print(f"    テーブル内からリンクを検索します...")
                                tables = soup.find_all('table')
                                for table in tables:
                                    # テーブル内のすべてのリンクを検索
                                    links_in_table = table.find_all('a', href=True)
                                    for link in links_in_table:
                                        href = link.get('href', '')
                                        link_text = link.get_text(strip=True)
                                        
                                        if debug_mode:
                                            print(f"      テーブル内リンク候補: text='{link_text}', href='{href}'")
                                        
                                        # file_download.php へのリンク、または .pos ファイル名を含むリンクを探す
                                        if ('file_download' in href.lower() or 
                                            filename.lower() in link_text.lower() or 
                                            '.pos' in link_text.lower()):
                                            download_link = href
                                            if debug_mode:
                                                print(f"    ✓ テーブル内でリンクを発見: {download_link}")
                                            break
                                    
                                    if download_link:
                                        break
                                    
                                    # リンクが見つからない場合は、セル内のテキストから探す（フォールバック）
                                    if not download_link:
                                        cells = table.find_all(['td', 'th'])
                                        for cell in cells:
                                            cell_text = cell.get_text(strip=True)
                                            # セル内にPOSファイル名が含まれているか確認
                                            if filename.lower() in cell_text.lower() or '.pos' in cell_text.lower():
                                                # セル内のリンクを探す（再帰的に）
                                                link = cell.find('a', href=True)
                                                if link:
                                                    download_link = link.get('href')
                                                    if debug_mode:
                                                        print(f"    ✓ セル内でリンクを発見: {download_link}")
                                                    break
                                        if download_link:
                                            break
                            
                            # リンクが見つかった場合は、そのリンクからファイルをダウンロード
                            if download_link:
                                # リンクが相対パスの場合は絶対URLに変換
                                if download_link.startswith('http'):
                                    file_url = download_link
                                elif download_link.startswith('/'):
                                    file_url = f"{GEONET_BASE_URL.rstrip('/')}{download_link}"
                                else:
                                    file_url = f"{GEONET_BASE_URL.rstrip('/')}/{download_link}"
                                
                                if debug_mode:
                                    print(f"    POSファイルをダウンロード: {file_url}")
                                
                                # ファイルをダウンロード
                                file_response = session.get(file_url, timeout=30)
                                if file_response.status_code == 200:
                                    file_content = file_response.text
                                    if file_content and len(file_content) > 100:
                                        if debug_mode:
                                            print(f"    ✓ POSファイルを取得: {filename} ({len(file_content)}文字)")
                                    else:
                                        if debug_mode:
                                            print(f"    ファイルが空または無効です（サイズ: {len(file_content)}文字）")
                                else:
                                    if debug_mode:
                                        print(f"    ファイルダウンロードエラー: ステータスコード {file_response.status_code}")
                            else:
                                # 「未登録」の場合はデータが存在しない
                                if '未登録' in file_content:
                                    if debug_mode:
                                        print(f"    ⚠ この観測地点・年のデータは未登録です")
                                else:
                                    if debug_mode:
                                        print(f"    ⚠ POSファイルへのリンクが見つかりませんでした。")
                        elif len(file_content) > 100:
                            if debug_mode:
                                print(f"    ✓ POST経由でファイルを取得: {filename} ({len(file_content)}文字)")
                        else:
                            if debug_mode:
                                print(f"    ファイルが空または無効です（サイズ: {len(file_content)}文字）")
                except requests.RequestException as e:
                    if debug_mode:
                        print(f"    POSTリクエストエラー: {e}")
                        import traceback
                        traceback.print_exc()
                except Exception as e:
                    if debug_mode:
                        print(f"    予期しないエラー: {e}")
                        import traceback
                        traceback.print_exc()
                        
        except requests.RequestException as e:
            if debug_mode:
                print(f"    Webエラー: {e}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            if debug_mode:
                print(f"    予期しないエラー: {e}")
                import traceback
                traceback.print_exc()
    
    if file_content:
        # POSファイルをパース
        year_data = parse_pos_file(file_content, station_id, station_info, year, debug_mode)
        all_data.extend(year_data)
        print(f"✓ {len(year_data)}件のデータを取得")
    else:
        print("✗ ファイルが見つかりませんでした")
    
    return all_data


def fetch_gnss_data(start_date=None, end_date=None, debug_mode=False):
    """
    GEONETからGPS・地殻変動データを取得
    
    Args:
        start_date: 開始日（デフォルト: 2010年1月1日）
        end_date: 終了日（デフォルト: 現在）
        debug_mode: デバッグモード
    
    Returns:
        list: GPSデータのリスト
    """
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = datetime.now()
    
    all_data = []
    
    print(f"データ取得を開始します...")
    print(f"開始日: {start_date.strftime('%Y-%m-%d')}")
    print(f"終了日: {end_date.strftime('%Y-%m-%d')}")
    print(f"観測地点数: {len(STATIONS)}")
    
    # 各観測地点についてデータを取得
    # GEONETのデータは年単位で提供されるため、年ごとに取得
    years = range(start_date.year, end_date.year + 1)
    
    for station_id, station_info in STATIONS.items():
        for year in years:
            year_data = fetch_gnss_data_for_year(
                station_id, station_info, year, debug_mode
            )
            all_data.extend(year_data)
            
            # レート制限を避けるため少し待機
            time.sleep(1)
        
        print()  # 観測地点ごとに改行
    
    print(f"\n合計 {len(all_data)} 件のデータを取得しました")
    return all_data


def get_latest_date_from_csv():
    """
    gnss_data.csvから最新の日付を取得
    
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


def load_existing_data():
    """既存のCSVファイルを読み込む"""
    if not CSV_FILE.exists():
        print(f"既存のファイルが見つかりません: {CSV_FILE}")
        return pd.DataFrame(columns=['date', 'station_id', 'station_name', 
                                     'latitude', 'longitude', 
                                     'east_mm', 'north_mm', 'up_mm'])
    
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"既存データ: {len(df)} 件")
        return df
    except Exception as e:
        print(f"既存データの読み込みエラー: {e}")
        return pd.DataFrame(columns=['date', 'station_id', 'station_name', 
                                     'latitude', 'longitude', 
                                     'east_mm', 'north_mm', 'up_mm'])


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
        key = (str(row['date']), str(row['station_id']))
        existing_set.add(key)
    
    # 新しいデータをフィルタリング
    new_records = []
    for record in new_data:
        key = (record['date'], record['station_id'])
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
    backup_file = work_folder / f"gnss_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(backup_file, index=False, encoding='utf-8-sig')
    print(f"バックアップを保存しました: {backup_file}")


def main():
    """メイン処理"""
    import sys
    
    # デバッグモードのチェック
    debug_mode = '--debug' in sys.argv or '-d' in sys.argv
    
    print("=" * 60)
    print("GPS・地殻変動データ自動取得スクリプト")
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
        latest_date = get_latest_date_from_csv()
        if latest_date:
            current_year = datetime.now().year
            current_date = datetime.now()
            
            print(f"既存データの最新日付: {latest_date.strftime('%Y-%m-%d')}")
            
            # 最新日付が現在年または前年の場合、その年のデータを再取得（最新データが追加されている可能性があるため）
            # GEONETのデータは年単位ファイルに含まれるため、最新日付が11月の場合でも12月のデータが追加されている可能性がある
            if latest_date.year == current_year or latest_date.year == current_year - 1:
                # 最新日付の年の1月1日から取得（年単位ファイルを再取得）
                target_year = latest_date.year
                start_date = datetime(target_year, 1, 1)
                print(f"取得開始日: {start_date.strftime('%Y-%m-%d')} ({target_year}年のデータを再取得)")
                if latest_date.year == current_year - 1:
                    print(f"  理由: 最新日付が前年({target_year}年)のため、最新データ（12月など）が追加されている可能性があります")
                else:
                    print(f"  理由: 最新日付が現在年({current_year}年)のため、最新データが追加されている可能性があります")
            else:
                # 既存データがある場合は、最新日付の翌年から取得
                # GNSSデータは年単位で取得されるため、既に取得済みの年はスキップ
                # 最新日付が2024年12月31日の場合、2025年1月1日から取得
                start_date = datetime(latest_date.year + 1, 1, 1)
                print(f"取得開始日: {start_date.strftime('%Y-%m-%d')} (最新日付の翌年から取得)")
                
                # 現在の年より未来の場合は、現在の年から取得
                if start_date.year > current_year:
                    start_date = datetime(current_year, 1, 1)
                    print(f"  注意: 取得開始年が未来のため、現在の年({current_year}年)から取得します")
        else:
            start_date = START_DATE
            print(f"既存データの最新日付を取得できませんでした")
            print(f"取得開始日: {start_date.strftime('%Y-%m-%d')} (デフォルト)")
    else:
        start_date = START_DATE
        print(f"既存データがありません")
        print(f"取得開始日: {start_date.strftime('%Y-%m-%d')} (デフォルト)")
    
    # デバッグモードでは直近1ヶ月のみ取得
    if debug_mode:
        start_date = datetime.now() - timedelta(days=30)
        print(f"デバッグモード: 直近1ヶ月のみ取得 ({start_date.strftime('%Y-%m-%d')})")
    
    # 4. 新しいデータを取得
    print("\n[データ取得]")
    new_data = fetch_gnss_data(start_date=start_date, debug_mode=debug_mode)
    
    if not new_data:
        print("新しいデータが取得できませんでした")
        if debug_mode:
            print("\n[デバッグ情報]")
            print(f"対象期間: {start_date.strftime('%Y-%m-%d')} から {datetime.now().strftime('%Y-%m-%d')}")
        
        # 新しいデータが取得できなくても、既存データを生データフォルダにコピー
        if not existing_df.empty:
            print("\n[生データフォルダへのコピー]")
            try:
                RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
                existing_df.to_csv(RAW_DATA_FILE, index=False, encoding='utf-8-sig')
                print(f"✓ 生データフォルダにコピーしました: {RAW_DATA_FILE}")
                print(f"  データ件数: {len(existing_df)} 件")
            except Exception as e:
                print(f"⚠ 生データフォルダへのコピーに失敗しました: {e}")
                print(f"  データは {CSV_FILE} に保存されています")
        
        print("\n" + "=" * 60)
        print("処理が完了しました")
        print("=" * 60)
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
            print(f"  データ件数: {len(existing_df)} 件")
        except Exception as e:
            print(f"⚠ 生データフォルダへのコピーに失敗しました: {e}")
            print(f"  データは {CSV_FILE} に保存されています")
        
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
    
    # 日付でソート（新しいデータが上に来るように）
    updated_df['date_parsed'] = pd.to_datetime(updated_df['date'], format='%Y/%m/%d', errors='coerce')
    updated_df = updated_df.sort_values('date_parsed', ascending=False)
    updated_df = updated_df.drop('date_parsed', axis=1)
    
    # CSVに保存
    updated_df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
    print(f"✓ データを追加しました: {CSV_FILE}")
    print(f"  既存: {len(existing_df)} 件")
    print(f"  追加: {len(new_df)} 件")
    print(f"  合計: {len(updated_df)} 件")
    
    # 8. 生データフォルダにもコピー
    print("\n[生データフォルダへのコピー]")
    try:
        RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        updated_df.to_csv(RAW_DATA_FILE, index=False, encoding='utf-8-sig')
        print(f"✓ 生データフォルダにコピーしました: {RAW_DATA_FILE}")
    except Exception as e:
        print(f"⚠ 生データフォルダへのコピーに失敗しました: {e}")
        print(f"  データは {CSV_FILE} に保存されています")
    
    # 9. 新しいデータをログファイルに保存
    log_file = work_folder / f"gnss_new_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    new_df.to_csv(log_file, index=False, encoding='utf-8-sig')
    print(f"✓ 新しいデータのログを保存しました: {log_file}")
    
    print("\n" + "=" * 60)
    print("処理が完了しました")
    print("=" * 60)


if __name__ == "__main__":
    main()

