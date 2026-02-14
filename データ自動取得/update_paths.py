"""
ファイル移動後のパス設定を更新するスクリプト
"""

from pathlib import Path
import re

def update_file_paths():
    """各スクリプトファイルのパス設定を更新"""
    base_dir = Path(__file__).parent
    src_dir = base_dir / "src"
    
    # 更新するファイルとパターン
    files_to_update = [
        {
            "file": src_dir / "fetch_earthquake_data.py",
            "old_pattern": r'CSV_FILE = DATA_DIR / "2010-all\.csv"',
            "new_pattern": 'CSV_FILE = DATA_DIR.parent / "data" / "2010-all.csv"'
        },
        {
            "file": src_dir / "fetch_world_earthquake_data.py",
            "old_pattern": r'CSV_FILE = DATA_DIR / "world-all\.csv"',
            "new_pattern": 'CSV_FILE = DATA_DIR.parent / "data" / "world-all.csv"'
        },
        {
            "file": src_dir / "fetch_depth_data.py",
            "old_pattern": r'CSV_FILE = DATA_DIR / "depth\.csv"',
            "new_pattern": 'CSV_FILE = DATA_DIR.parent / "data" / "depth.csv"'
        },
        {
            "file": src_dir / "fetch_depth_data.py",
            "old_pattern": r'debug_file = DATA_DIR / f"debug_html_',
            "new_pattern": 'debug_file = DATA_DIR.parent / "debug" / f"debug_html_'
        }
    ]
    
    for item in files_to_update:
        file_path = item["file"]
        if not file_path.exists():
            print(f"⚠ ファイルが見つかりません: {file_path}")
            continue
        
        try:
            # ファイルを読み込む
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # パターンを置換
            old_content = content
            content = re.sub(item["old_pattern"], item["new_pattern"], content)
            
            if content != old_content:
                # ファイルに書き戻す
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✓ 更新: {file_path.name}")
            else:
                print(f"  (変更なし): {file_path.name}")
        except Exception as e:
            print(f"✗ エラー ({file_path.name}): {e}")
    
    print("\nパス設定の更新が完了しました！")

if __name__ == "__main__":
    update_file_paths()





















