"""
world-all.csvのデータを正規化するスクリプト
USGS APIのplaceフィールドから国名を抽出して、既存データを修正します。
"""

import pandas as pd
from pathlib import Path
import sys

# 設定
DATA_DIR = Path(__file__).parent
CSV_FILE = DATA_DIR.parent / "data" / "world-all.csv"
PROJECT_ROOT = DATA_DIR.parent.parent
RAW_DATA_FILE = PROJECT_ROOT / "生データ" / "world-all.csv"

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
    if not place or pd.isna(place) or str(place) == 'nan':
        return place
    
    place = str(place).strip()
    
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


def main():
    """メイン処理"""
    print("=" * 60)
    print("world-all.csvデータ正規化スクリプト")
    print("=" * 60)
    
    # ファイルを読み込む
    files_to_process = []
    if CSV_FILE.exists():
        files_to_process.append(CSV_FILE)
    if RAW_DATA_FILE.exists():
        files_to_process.append(RAW_DATA_FILE)
    
    if not files_to_process:
        print("処理するファイルが見つかりません")
        return
    
    for file_path in files_to_process:
        print(f"\n処理中: {file_path}")
        
        try:
            # CSVファイルを読み込む
            df = pd.read_csv(file_path)
            print(f"  読み込み: {len(df)} 件")
            
            # バックアップを作成
            backup_file = file_path.parent / f"{file_path.stem}_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(backup_file, index=False, encoding='utf-8-sig')
            print(f"  バックアップ作成: {backup_file}")
            
            # place_wカラムを正規化
            if 'place_w' in df.columns:
                original_places = df['place_w'].copy()
                df['place_w'] = df['place_w'].apply(extract_country_name)
                
                # 変更があった行を確認
                changed_rows = df[original_places != df['place_w']]
                if not changed_rows.empty:
                    print(f"  修正した行数: {len(changed_rows)} 件")
                    if len(changed_rows) <= 10:
                        print("  修正内容:")
                        for idx, row in changed_rows.iterrows():
                            print(f"    {row['date']}: '{original_places.iloc[idx]}' → '{row['place_w']}'")
                    else:
                        print("  修正内容（最初の10件）:")
                        for idx, row in changed_rows.head(10).iterrows():
                            print(f"    {row['date']}: '{original_places.iloc[idx]}' → '{row['place_w']}'")
                else:
                    print("  修正が必要な行はありませんでした")
            else:
                print("  'place_w'カラムが見つかりません")
                continue
            
            # CSVファイルに保存
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"  保存完了: {file_path}")
            
        except Exception as e:
            print(f"  エラー: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("処理が完了しました")
    print("=" * 60)


if __name__ == "__main__":
    main()
