"""
深さデータ自動取得スクリプト
F-net（防災科研）から深さデータを取得し、depth.csvに追加します。
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime, timedelta
from pathlib import Path
import time
import sys
from dateutil.relativedelta import relativedelta

# Seleniumのインポート（オプション、JavaScriptが必要な場合）
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# 設定
BASE_URL = "https://www.fnet.bosai.go.jp/event/joho.php"
DATA_DIR = Path(__file__).parent
CSV_FILE = DATA_DIR.parent / "data" / "depth.csv"
# プロジェクトルートの生データフォルダへのパス
PROJECT_ROOT = DATA_DIR.parent.parent
RAW_DATA_FILE = PROJECT_ROOT / "生データ" / "depth.csv"
MIN_DEPTH = 300.0  # 最小深さ（km）
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def get_recent_months(existing_df=None, months_back=6):
    """
    直近N ヶ月の年月を取得（過去データの後追い登録に対応）

    F-netサイトでは過去データが後から追加登録されることがあるため、
    デフォルトで直近6ヶ月分を毎回取得し、重複は除外する方式を採用。

    Args:
        existing_df: 既存のDataFrame（オプション）
        months_back: 遡る月数（デフォルト: 6ヶ月）

    Returns:
        list: [(year, month), ...] の形式のリスト（新しい順）
    """
    today = datetime.now()
    months_set = set()

    # 直近N ヶ月を取得（過去データの後追い登録に対応）
    for i in range(months_back):
        target_date = today - relativedelta(months=i)
        months_set.add((target_date.year, target_date.month))

    print(f"[情報] 直近{months_back}ヶ月分を取得対象とします（過去データの後追い登録に対応）")

    # 既存データがある場合、統計情報を表示
    if existing_df is not None and not existing_df.empty:
        try:
            existing_df_copy = existing_df.copy()
            existing_df_copy['date_parsed'] = pd.to_datetime(existing_df_copy['date'], format='%Y/%m/%d', errors='coerce')
            latest_date = existing_df_copy['date_parsed'].max()
            oldest_date = existing_df_copy['date_parsed'].min()
            if pd.notna(latest_date) and pd.notna(oldest_date):
                print(f"[情報] 既存データの範囲: {oldest_date.strftime('%Y/%m/%d')} ～ {latest_date.strftime('%Y/%m/%d')}")
        except Exception as e:
            print(f"[警告] 既存データの情報を取得できませんでした: {e}")

    # セットをリストに変換してソート（新しい順）
    months = sorted(months_set, key=lambda x: (x[0], x[1]), reverse=True)
    return months


def fetch_depth_data_for_month(year, month, debug_mode=False, use_selenium=False):
    """
    指定された年月の深さデータを取得
    
    Args:
        year: 年
        month: 月
        debug_mode: デバッグモード
        use_selenium: Seleniumを使用するかどうか
    
    Returns:
        list: 地震データのリスト [{'date': '2025/12/20', 'depth': 350.5}, ...]
    """
    all_data = []
    
    print(f"  {year}年{month}月のデータを取得中...", end=" ", flush=True)
    
    # F-netのURLパラメータ（GETリクエスト用）
    params = {
        'LANG': 'ja',
        'year': year,
        'month': month
    }
    
    # POSTリクエスト用のデータ（フォーム送信）
    post_data = {
        'LANG': 'ja',
        'year': str(year),
        'month': str(month),
        'display': 'all'  # 全てを表示
    }
    
    if debug_mode:
        print(f"\n    リクエストURL: {BASE_URL}")
        print(f"    GETパラメータ: {params}")
        print(f"    POSTデータ: {post_data}")
        print(f"    Selenium使用: {use_selenium}")
    
    html_content = None
    response = None  # デバッグ用に初期化
    
    # Seleniumを使用する場合
    if use_selenium and SELENIUM_AVAILABLE:
        try:
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            chrome_options = Options()
            chrome_options.add_argument('--headless=new')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument(f'user-agent={USER_AGENT}')

            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(60)

            # まずベースURLにアクセス
            url = f"{BASE_URL}?LANG=ja"
            driver.get(url)

            # 初期ページの読み込みを待機
            wait = WebDriverWait(driver, 30)
            time.sleep(3)

            # F-netサイトはpageViewer関数で月を切り替える
            # 形式: pageViewer('YYYYMM', 'ja', false)
            year_month = f"{year}{month:02d}"
            js_command = f"pageViewer('{year_month}', 'ja', false)"
            if debug_mode:
                print(f"\n    JavaScript実行: {js_command}")
            driver.execute_script(js_command)

            # データ読み込みを待機
            time.sleep(8)

            # テーブルが読み込まれるまで追加待機
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'table.sret_02')))
                wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, 'table.sret_02 tr')) > 3)
                time.sleep(2)

                if debug_mode:
                    row_count = len(driver.find_elements(By.CSS_SELECTOR, 'table.sret_02 tr'))
                    print(f"    テーブル行数: {row_count}")

            except Exception as e:
                if debug_mode:
                    print(f"\n    テーブル待機エラー: {e}")
                time.sleep(5)

            html_content = driver.page_source
            driver.quit()

        except Exception as e:
            print(f"\n    Seleniumエラー: {e}")
            if debug_mode:
                import traceback
                traceback.print_exc()
            # Seleniumが失敗した場合は通常のrequestsにフォールバック
            use_selenium = False
            html_content = None
    
    # requestsを使用する場合（デフォルトまたはSelenium失敗時）
    if html_content is None:
        session = requests.Session()
        session.headers.update({'User-Agent': USER_AGENT})
        
        try:
            # POSTリクエストを試す（フォーム送信）
            response = session.post(BASE_URL, data=post_data, timeout=30)
            # F-netサイトはEUC-JPエンコーディングの可能性がある
            if 'charset=euc-jp' in response.headers.get('content-type', '').lower() or 'euc-jp' in response.apparent_encoding.lower():
                response.encoding = 'euc-jp'
            else:
                response.encoding = 'utf-8'
            response.raise_for_status()
            
            if debug_mode:
                print(f"    レスポンスステータス: {response.status_code}")
                print(f"    レスポンスサイズ: {len(response.text)} 文字")
                print(f"    エンコーディング: {response.encoding}")
            
            html_content = response.text
            
        except requests.RequestException as e:
            print(f"✗ POSTエラー: {e}")
            # POSTが失敗した場合はGETを試す
            try:
                response = session.get(BASE_URL, params=params, timeout=30)
                if 'charset=euc-jp' in response.headers.get('content-type', '').lower() or 'euc-jp' in response.apparent_encoding.lower():
                    response.encoding = 'euc-jp'
                else:
                    response.encoding = 'utf-8'
                response.raise_for_status()
                html_content = response.text
                if debug_mode:
                    print(f"    GETリクエストで取得しました")
            except requests.RequestException as e2:
                print(f"✗ GETエラー: {e2}")
                return []
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # HTMLをファイルに保存（デバッグ用）
        if debug_mode:
            debug_file = DATA_DIR.parent / "debug" / f"debug_html_{year}_{month}.html"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"\n    HTMLを保存しました: {debug_file}")
            # HTMLの一部を表示
            html_sample = str(soup)[:3000]
            print(f"\n    HTMLサンプル（最初の3000文字）: {html_sample}")
        
        # テーブルを探す
        tables = soup.find_all('table')
        # tbody内のテーブルも探す（tbodyが独立している場合）
        tbody_tables = soup.find_all('tbody')
        
        # tbody内にテーブルがある場合、それも含める
        for tbody in tbody_tables:
            tbody_tables_in_tbody = tbody.find_all('table')
            if tbody_tables_in_tbody:
                tables.extend(tbody_tables_in_tbody)
        
        print(f"    見つかったテーブル数: {len(tables)}")
        print(f"    見つかったtbody数: {len(tbody_tables)}")
        
        if not tables:
            print("テーブルが見つかりません")
            # ページの構造を確認
            divs = soup.find_all('div')
            print(f"    見つかったdiv数: {len(divs)}")
            # データが含まれていそうなdivを探す
            data_divs = []
            for div in divs:
                div_class = div.get('class', [])
                div_id = div.get('id', '')
                div_text = div.get_text(strip=True)[:100]
                if any(keyword in str(div_class).lower() or keyword in str(div_id).lower() 
                       for keyword in ['table', 'data', 'result', 'event', 'earthquake']):
                    data_divs.append((div_class, div_id, div_text))
            if data_divs:
                print(f"    データ関連のdivを発見: {len(data_divs)}個")
                for i, (cls, id_attr, text) in enumerate(data_divs[:5], 1):
                    print(f"      div{i}: class={cls}, id={id_attr}, text={text[:50]}")
            
            # HTMLをファイルに保存（デバッグ用）
            debug_html_file = DATA_DIR.parent / "debug" / f"debug_html_no_table_{year}_{month}.html"
            debug_html_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(debug_html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                print(f"    HTMLを保存しました: {debug_html_file}")
            except Exception as e:
                print(f"    HTMLの保存に失敗しました: {e}")
            
            return []
        
        # データテーブルを探す（class="sret_02"のテーブルを優先的に探す）
        table = None
        
        # まず、class="sret_02"のテーブルを探す
        for t in tables:
            if t.get('class') and 'sret_02' in t.get('class'):
                table = t
                if debug_mode:
                    print(f"    class='sret_02'のテーブルを発見")
                break
        
        # sret_02が見つからない場合、データ行が多いテーブルを探す
        if not table:
            print(f"    class='sret_02'のテーブルが見つかりません。データ行が多いテーブルを探します...")
            for i, t in enumerate(tables):
                rows = t.find_all('tr', recursive=True)
                # データ行が5行以上あるテーブルを探す（10行から5行に緩和）
                if len(rows) >= 5:
                    # テーブルにデータが含まれているかを確認（日付や深さのキーワードが含まれているか）
                    table_text = t.get_text().lower()
                    if any(keyword in table_text for keyword in ['深さ', 'depth', '日時', 'date', '時刻', 'time', '発生']):
                        table = t
                        print(f"    データテーブルとして選択: テーブル{i+1} ({len(rows)}行)")
                        break
                    elif len(rows) >= 10:
                        # キーワードが見つからなくても、10行以上あればデータテーブルとみなす
                        table = t
                        print(f"    データテーブルとして選択: テーブル{i+1} ({len(rows)}行) [キーワードなし]")
                        break
        
        if not table:
            print("データテーブルが見つかりません")
            # デバッグ情報を常に出力（テーブルが見つからない場合は重要）
            print(f"    見つかったテーブル数: {len(tables)}")
            if tables:
                print("    見つかったテーブルの詳細:")
                for i, t in enumerate(tables[:5], 1):  # 最初の5つを表示
                    rows = t.find_all('tr', recursive=True)
                    class_attr = t.get('class', [])
                    print(f"      テーブル{i}: 行数={len(rows)}, class={class_attr}")
                    if rows:
                        first_row_text = rows[0].get_text(strip=True)[:100]
                        print(f"        最初の行: {first_row_text}")
            else:
                print("    テーブルが1つも見つかりませんでした")
                # HTMLの構造を確認
                print("    HTMLの構造を確認中...")
                divs = soup.find_all('div')
                print(f"    div要素数: {len(divs)}")
                # データが含まれていそうなdivを探す
                for div in divs[:10]:  # 最初の10個を確認
                    div_class = div.get('class', [])
                    div_id = div.get('id', '')
                    if 'table' in str(div_class).lower() or 'data' in str(div_class).lower() or 'result' in str(div_class).lower():
                        print(f"      データ関連のdivを発見: class={div_class}, id={div_id}")
            
            # HTMLをファイルに保存（デバッグ用）
            debug_html_file = DATA_DIR.parent / "debug" / f"debug_html_no_table_{year}_{month}.html"
            debug_html_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(debug_html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                print(f"    HTMLを保存しました: {debug_html_file}")
            except Exception as e:
                print(f"    HTMLの保存に失敗しました: {e}")
            
            return []
        
        # テーブルの行を取得（再帰的に検索、tbody内も含む）
        rows = table.find_all('tr', recursive=True)
        if debug_mode:
            print(f"    テーブルの行数: {len(rows)}")
            if rows:
                first_row_text = rows[0].get_text(strip=True)[:200]
                print(f"    最初の行の内容: {first_row_text}")
                if len(rows) > 1:
                    second_row_text = rows[1].get_text(strip=True)[:200]
                    print(f"    2行目の内容: {second_row_text}")
        
        if len(rows) <= 1:
            print("データが見つかりません")
            return []
        
        # F-netサイトのテーブル構造に基づいてカラム位置を特定
        # データ行の構造: No, 日時, 緯度, 経度, 震央地名, 深さ(気象庁), Mj, 深さ(MT解), Mw, 品質, ボタン
        # 深さは2つあるが、MT解推定による深さ（7番目のtd、インデックス7）を使用
        # 日時は2番目のtd（インデックス1）のspan内
        
        # ヘッダー行を確認（2行のヘッダーがある可能性がある）
        header_rows = []
        for i, row in enumerate(rows[:5]):  # 最初の5行を確認
            cells = row.find_all(['th', 'td'])
            row_text = row.get_text(strip=True).lower()
            # ヘッダー行の特徴: 「深さ」「日時」「no」などのキーワードが含まれる
            if any(keyword in row_text for keyword in ['深さ', 'depth', '日時', 'date', '時刻', 'time', '発生', 'no', '気象庁', 'mt解']):
                header_rows.append(i)
                if debug_mode:
                    print(f"    ヘッダー行{i+1}を発見: {row_text[:100]}")
        
        # データ行の開始位置を特定（ヘッダー行の次から）
        data_start_idx = max(header_rows) + 1 if header_rows else 2  # デフォルトは2（2行のヘッダーを想定）
        
        if debug_mode:
            print(f"    データ行の開始インデックス: {data_start_idx}")
        
        # データ行を処理
        data_rows = rows[data_start_idx:]
        
        month_data_count = 0
        processed_rows = 0
        
        # 12月26日と27日のデータを探すためのフラグ
        found_dec_26_27_in_html = False
        
        for row_idx, row in enumerate(data_rows):
            cells = row.find_all('td')  # thは除外
            if len(cells) < 8:  # 最低限のカラム数が必要（日付と深さを含む）
                continue
            
            processed_rows += 1
            if debug_mode and row_idx < 5:
                cell_texts = [cell.get_text(strip=True) for cell in cells[:8]]
                print(f"    行{row_idx+1}のセル数: {len(cells)}, 内容: {cell_texts}")
            
            try:
                date_text = ""
                depth_value = None
                
                # 日時は2番目のtd（インデックス1）のspan内
                if len(cells) > 1:
                    date_cell = cells[1]
                    # spanタグ内のテキストを取得
                    span = date_cell.find('span')
                    if span:
                        date_cell_text = span.get_text(strip=True)
                    else:
                        date_cell_text = date_cell.get_text(strip=True)
                    
                    # 12月26日と27日のデータがHTMLに含まれているかを確認
                    if not found_dec_26_27_in_html and ('12/26' in date_cell_text or '12/27' in date_cell_text or '12月26' in date_cell_text or '12月27' in date_cell_text):
                        found_dec_26_27_in_html = True
                        print(f"      [重要] HTMLに12月26日または27日のデータが含まれていることを確認しました")
                        print(f"      該当セルの内容: {date_cell_text[:100]}")
                    
                    # YYYY/MM/DD形式を抽出（例: "2025/12/16,23:38:40.31" -> "2025/12/16"）
                    # カンマやスペースの前の日付部分を抽出
                    # 複数のパターンを試す
                    date_match = None
                    # パターン1: "YYYY/MM/DD,HH:MM:SS.ss" 形式
                    date_match = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})[, ]', date_cell_text)
                    if not date_match:
                        # パターン2: "YYYY/MM/DD" 形式（時刻なし）
                        date_match = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?!\d)', date_cell_text)
                    if date_match:
                        year_part = date_match.group(1)
                        month_part = date_match.group(2).zfill(2)
                        day_part = date_match.group(3).zfill(2)
                        date_text = f"{year_part}/{month_part}/{day_part}"
                    elif debug_mode:
                        # デバッグモードの場合、抽出できなかった日付セルの内容を表示
                        print(f"      日付抽出失敗: {date_cell_text[:100]}")
                        # 12月26日と27日のデータを探している場合、特に詳細にログ出力
                        if '12/26' in date_cell_text or '12/27' in date_cell_text or '12月26' in date_cell_text or '12月27' in date_cell_text:
                            print(f"      [重要] 12月26日または27日のデータが見つかりましたが、抽出に失敗しました")
                            print(f"      セルの完全な内容: {date_cell_text}")
                
                # MT解推定による深さは7番目のtd（インデックス7）
                # ただし、気象庁による深さ（インデックス5）も確認し、大きい方を優先
                depth_values = []
                
                # 気象庁による深さ（インデックス5）
                if len(cells) > 5:
                    depth_cell_text = cells[5].get_text(strip=True)
                    # 数値を抽出（小数点を含む可能性がある）
                    depth_match = re.search(r'(\d+\.?\d*)', depth_cell_text)
                    if depth_match:
                        try:
                            depth_val = float(depth_match.group(1))
                            if depth_val >= MIN_DEPTH:
                                depth_values.append(depth_val)
                                if debug_mode and date_text in ['2025/12/26', '2025/12/27']:
                                    print(f"      [重要] 気象庁深さを発見: {date_text}, {depth_val}km (セル5)")
                        except ValueError:
                            pass
                
                # MT解推定による深さ（インデックス7）
                if len(cells) > 7:
                    depth_cell_text = cells[7].get_text(strip=True)
                    # 数値を抽出（小数点を含む可能性がある）
                    depth_match = re.search(r'(\d+\.?\d*)', depth_cell_text)
                    if depth_match:
                        try:
                            depth_val = float(depth_match.group(1))
                            if depth_val >= MIN_DEPTH:
                                depth_values.append(depth_val)
                                if debug_mode and date_text in ['2025/12/26', '2025/12/27']:
                                    print(f"      [重要] MT解深さを発見: {date_text}, {depth_val}km (セル7)")
                        except ValueError:
                            pass
                
                # デバッグモードで12月26日と27日のデータを探している場合、全てのセルの内容を表示
                if debug_mode and date_text in ['2025/12/26', '2025/12/27']:
                    print(f"      [重要] 12月26日または27日のデータを処理中:")
                    print(f"        日付: {date_text}")
                    print(f"        セル数: {len(cells)}")
                    for i, cell in enumerate(cells[:10]):  # 最初の10セルを表示
                        cell_text = cell.get_text(strip=True)
                        print(f"        セル{i}: {cell_text[:50]}")
                    print(f"        抽出された深さ値: {depth_values}")
                
                # 深さが300km以上の場合は、最大値を使用
                if depth_values:
                    depth_value = max(depth_values)
                
                # 日付と深さの両方が見つかり、深さが300km以上の場合のみ追加
                if date_text and depth_value is not None and depth_value >= MIN_DEPTH:
                    all_data.append({
                        'date': date_text,
                        'depth': depth_value
                    })
                    month_data_count += 1
                    if debug_mode and month_data_count <= 3:
                        print(f"      データ追加: {date_text}, {depth_value}km")
                    # 12月26日と27日のデータを特別にログ出力
                    if date_text in ['2025/12/26', '2025/12/27']:
                        print(f"      [重要] {date_text}, {depth_value}km を取得しました")
                elif debug_mode:
                    # データが追加されなかった理由をログ出力
                    if not date_text:
                        print(f"      日付が抽出できませんでした: {date_cell_text[:50]}")
                    elif depth_value is None:
                        print(f"      深さが抽出できませんでした: 日付={date_text}")
                    elif depth_value < MIN_DEPTH:
                        print(f"      深さが{MIN_DEPTH}km未満です: {date_text}, {depth_value}km")
                    
            except Exception as e:
                if debug_mode:
                    print(f"\n    データ処理エラー: {e}")
                continue
        
        if debug_mode:
            print(f"    処理した行数: {processed_rows}")
        print(f"✓ {month_data_count}件のデータを取得")
        
        # 12月26日と27日のデータがHTMLに含まれていたが取得できなかった場合に警告
        if found_dec_26_27_in_html and year == 2025 and month == 12:
            dec_26_27_extracted = [r for r in all_data if r.get('date', '') in ['2025/12/26', '2025/12/27']]
            if not dec_26_27_extracted:
                print(f"      [警告] HTMLに12月26日または27日のデータが含まれていましたが、抽出できませんでした")
                print(f"      データ抽出ロジックを確認してください")
        
    except Exception as e:
        print(f"✗ 予期しないエラー: {e}")
        if debug_mode:
            import traceback
            traceback.print_exc()
        return []
    
    return all_data


def fetch_depth_data(debug_mode=False, use_selenium=False, existing_df=None):
    """
    F-netから深さデータを取得（直近6ヶ月分を毎回取得）

    F-netサイトでは過去データが後から追加登録されることがあるため、
    直近6ヶ月分を毎回取得し、重複は除外する方式を採用。

    Args:
        debug_mode: デバッグモード
        use_selenium: Seleniumを使用するかどうか
        existing_df: 既存のDataFrame（統計情報表示用）

    Returns:
        list: 地震データのリスト [{'date': '2025/12/20', 'depth': 350.5}, ...]
    """
    all_data = []

    print(f"データ取得を開始します...")
    print(f"URL: {BASE_URL}")
    print(f"対象: 直近6ヶ月（過去データの後追い登録に対応）")
    print(f"最小深さ: {MIN_DEPTH}km")
    
    if use_selenium and not SELENIUM_AVAILABLE:
        print("=" * 50)
        print("警告: Seleniumが利用できません！")
        print("F-netサイトはJavaScriptでデータを読み込むため、")
        print("Seleniumなしでは正確なデータ取得ができません。")
        print()
        print("Seleniumをインストールするには:")
        print("  pip install selenium")
        print("  pip install webdriver-manager")
        print("=" * 50)
        use_selenium = False

    # 直近6ヶ月を取得（過去データの後追い登録に対応）
    months = get_recent_months(existing_df)
    print(f"取得対象月: {months}")

    for year, month in months:
        print(f"\n[{year}年{month}月のデータを取得中]")
        month_data = fetch_depth_data_for_month(year, month, debug_mode, use_selenium)
        all_data.extend(month_data)

        # レート制限を避けるため少し待機
        time.sleep(1)

    print(f"\n合計 {len(all_data)} 件のデータを取得しました")

    # 取得データの日付範囲を表示
    if all_data:
        dates = [r.get('date', '') for r in all_data if r.get('date')]
        if dates:
            print(f"取得データの日付範囲: {min(dates)} ～ {max(dates)}")
    
    return all_data


def load_existing_data():
    """既存のCSVファイルを読み込む"""
    if not CSV_FILE.exists():
        print(f"既存のファイルが見つかりません: {CSV_FILE}")
        return pd.DataFrame(columns=['date', 'depth'])
    
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"既存データ: {len(df)} 件")
        return df
    except Exception as e:
        print(f"既存データの読み込みエラー: {e}")
        return pd.DataFrame(columns=['date', 'depth'])


def find_new_data(existing_df, new_data):
    """
    新しいデータを特定（既存データとの重複、および取得データ内の重複を除去）
    
    Args:
        existing_df: 既存のDataFrame
        new_data: 新しく取得したデータのリスト
    
    Returns:
        DataFrame: 新しいデータのみを含むDataFrame
    """
    # 既存データのセットを作成（重複チェック用）
    existing_set = set()
    if not existing_df.empty:
        for _, row in existing_df.iterrows():
            # 日付を正規化（YYYY/MM/DD形式に統一）
            date_str = str(row['date']).strip()
            depth = float(row['depth'])
            key = (date_str, depth)
            existing_set.add(key)
    
    # 新しいデータをフィルタリング（既存データとの重複、および取得データ内の重複を除去）
    new_records = []
    seen_keys = set()  # 取得データ内の重複チェック用
    excluded_by_existing = []  # 既存データとの重複で除外されたデータ
    excluded_by_duplicate = []  # 取得データ内の重複で除外されたデータ
    
    for record in new_data:
        key = (record['date'], record['depth'])
        # 既存データとの重複チェック
        if key not in existing_set:
            # 取得データ内の重複チェック
            if key not in seen_keys:
                new_records.append(record)
                seen_keys.add(key)
            else:
                excluded_by_duplicate.append(record)
        else:
            excluded_by_existing.append(record)
    
    # 12月26日と27日のデータが除外された場合に警告
    dec_26_27_excluded = [r for r in excluded_by_existing if '2025/12/26' in r.get('date', '') or '2025/12/27' in r.get('date', '')]
    if dec_26_27_excluded:
        print(f"\n[警告] 12月26日と27日のデータが既存データとの重複で除外されました:")
        for record in dec_26_27_excluded:
            print(f"  除外: {record}")
            # 既存データに同じデータが存在するかを確認
            if not existing_df.empty:
                matching_rows = existing_df[
                    (existing_df['date'].astype(str).str.contains(record['date'], na=False)) &
                    (existing_df['depth'].astype(float) == record['depth'])
                ]
                if not matching_rows.empty:
                    print(f"    既存データに同じデータが存在します:")
                    for _, row in matching_rows.iterrows():
                        print(f"      既存: {row['date']}, {row['depth']}km")
                else:
                    print(f"    [注意] 既存データに完全一致するデータが見つかりませんでした（日付または深さの形式が異なる可能性があります）")
    
    # 12月26日と27日のデータが新しいデータとして追加される場合に確認
    dec_26_27_new = [r for r in new_records if '2025/12/26' in r.get('date', '') or '2025/12/27' in r.get('date', '')]
    if dec_26_27_new:
        print(f"\n[確認] 12月26日と27日のデータが新しいデータとして追加されます:")
        for record in dec_26_27_new:
            print(f"  追加: {record}")
    
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
    backup_file = work_folder / f"depth_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(backup_file, index=False, encoding='utf-8-sig')
    print(f"バックアップを保存しました: {backup_file}")


def main():
    """メイン処理"""
    import sys
    
    # デバッグモードのチェック
    debug_mode = '--debug' in sys.argv or '-d' in sys.argv
    # Selenium使用のチェック
    use_selenium = '--selenium' in sys.argv or '-s' in sys.argv
    
    print("=" * 60)
    print("深さデータ自動取得スクリプト")
    print("=" * 60)
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"データファイル: {CSV_FILE}")
    if debug_mode:
        print("デバッグモード: ON")
    if use_selenium:
        print("Seleniumモード: ON")
    print("-" * 60)
    
    # 1. 作業フォルダを作成
    work_folder = create_work_folder()
    
    # 2. 既存データを読み込む
    print("\n[既存データの読み込み]")
    existing_df = load_existing_data()
    
    # 3. 新しいデータを取得
    print("\n[データ取得]")
    # F-netサイトはJavaScriptで動的にデータを読み込むため、Selenium必須
    if SELENIUM_AVAILABLE:
        print("Seleniumを使用してデータを取得します。")
        use_selenium = True
    else:
        print("=" * 60)
        print("エラー: Seleniumが利用できません！")
        print()
        print("F-netサイトはJavaScriptでデータを動的に読み込むため、")
        print("Seleniumがないと正確なデータ取得ができません。")
        print()
        print("以下のコマンドでSeleniumをインストールしてください:")
        print("  pip install selenium webdriver-manager")
        print("=" * 60)
        print()
        print("requestsでの取得を試みますが、データが不完全な可能性があります...")
    
    # 既存データを渡して、最新日付の月も取得対象に含める
    new_data = fetch_depth_data(debug_mode=debug_mode, use_selenium=use_selenium, existing_df=existing_df)
    
    if not new_data:
        print("新しいデータが取得できませんでした")
        if debug_mode:
            print("\n[デバッグ情報]")
            months = get_recent_months(existing_df)
            print(f"対象月: {months}")
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
                print("取得したデータのサンプル（最初の10件）:")
                for i, record in enumerate(new_data[:10], 1):
                    print(f"  {i}. {record}")
                # 12月26日と27日のデータを検索
                print("\n[12月26日と27日のデータ検索]")
                dec_26_27_data = [r for r in new_data if '2025/12/26' in r.get('date', '') or '2025/12/27' in r.get('date', '')]
                if dec_26_27_data:
                    print(f"見つかったデータ: {len(dec_26_27_data)}件")
                    for record in dec_26_27_data:
                        print(f"  {record}")
                else:
                    print("12月26日と27日のデータは見つかりませんでした")
        else:
            # デバッグモードでなくても、12月26日と27日のデータを確認
            dec_26_27_data = [r for r in new_data if '2025/12/26' in r.get('date', '') or '2025/12/27' in r.get('date', '')]
            if dec_26_27_data:
                print(f"\n[注意] 12月26日と27日のデータが取得されましたが、重複チェックで除外された可能性があります")
                print(f"取得されたデータ: {dec_26_27_data}")
            else:
                print(f"\n[注意] 12月26日と27日のデータが取得されていません（取得データ数: {len(new_data)}件）")
                if new_data:
                    print("取得されたデータの日付範囲:")
                    dates = [r.get('date', '') for r in new_data]
                    if dates:
                        print(f"  最初の日付: {min(dates)}")
                        print(f"  最後の日付: {max(dates)}")
        
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
    
    # 4-1. 12月26日と27日のデータが新しいデータに含まれているかを確認
    dec_26_27_in_new = new_df[
        (new_df['date'].astype(str).str.contains('2025/12/26', na=False)) |
        (new_df['date'].astype(str).str.contains('2025/12/27', na=False))
    ]
    if not dec_26_27_in_new.empty:
        print(f"\n[確認] 12月26日と27日のデータが新しいデータとして追加されます:")
        for _, row in dec_26_27_in_new.iterrows():
            print(f"  {row['date']}, {row['depth']}km")
    else:
        print(f"\n[警告] 12月26日と27日のデータが新しいデータに含まれていません")
        # 取得データには含まれていたが、重複チェックで除外された可能性を確認
        dec_26_27_in_fetched = [r for r in new_data if '2025/12/26' in r.get('date', '') or '2025/12/27' in r.get('date', '')]
        if dec_26_27_in_fetched:
            print(f"  取得データには含まれていましたが、重複チェックで除外された可能性があります")
            print(f"  取得データ: {dec_26_27_in_fetched}")
        else:
            print(f"  取得データにも含まれていませんでした（HTMLから取得できなかった可能性があります）")
    
    # 5. バックアップを保存
    print("\n[バックアップ作成]")
    save_backup(existing_df, work_folder)
    
    # 6. データを追加
    print("\n[データ追加]")
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    
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
    
    # 6-1. 生データフォルダにもコピー
    print("\n[生データフォルダへのコピー]")
    try:
        # 生データフォルダが存在しない場合は作成
        RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        updated_df.to_csv(RAW_DATA_FILE, index=False, encoding='utf-8-sig')
        print(f"✓ 生データフォルダにコピーしました: {RAW_DATA_FILE}")
    except Exception as e:
        print(f"⚠ 生データフォルダへのコピーに失敗しました: {e}")
        print(f"  データは {CSV_FILE} に保存されています")
    
    # 7. 新しいデータをログファイルに保存
    log_file = work_folder / f"depth_new_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    new_df.to_csv(log_file, index=False, encoding='utf-8-sig')
    print(f"✓ 新しいデータのログを保存しました: {log_file}")
    
    # 7-1. 最終確認: 12月26日と27日のデータがCSVに含まれているかを確認
    final_dec_26_27 = updated_df[
        (updated_df['date'].astype(str).str.contains('2025/12/26', na=False)) |
        (updated_df['date'].astype(str).str.contains('2025/12/27', na=False))
    ]
    if not final_dec_26_27.empty:
        print(f"\n[最終確認] 12月26日と27日のデータがCSVに含まれています:")
        for _, row in final_dec_26_27.iterrows():
            print(f"  {row['date']}, {row['depth']}km")
    else:
        print(f"\n[最終確認] 12月26日と27日のデータがCSVに含まれていません")
        print(f"  既存データを確認してください: {RAW_DATA_FILE}")
    
    print("\n" + "=" * 60)
    print("処理が完了しました")
    print("=" * 60)


if __name__ == "__main__":
    main()
