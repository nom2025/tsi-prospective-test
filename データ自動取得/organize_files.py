"""
データ自動取得フォルダの整理スクリプト
ファイルを適切なフォルダに移動します。
"""

import shutil
import os
from pathlib import Path

def organize_files():
    """ファイルを整理する"""
    # 現在のスクリプトのディレクトリを取得
    script_path = Path(__file__).resolve()
    base_dir = script_path.parent
    
    print(f"作業ディレクトリ: {base_dir}")
    
    # フォルダを作成
    src_dir = base_dir / "src"
    data_dir = base_dir / "data"
    debug_dir = base_dir / "debug"
    
    src_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    debug_dir.mkdir(exist_ok=True)
    
    print("フォルダを作成しました")
    
    # ソースファイルを移動
    source_files = [
        "fetch_earthquake_data.py",
        "fetch_world_earthquake_data.py",
        "fetch_depth_data.py"
    ]
    
    for file_name in source_files:
        src_file = base_dir / file_name
        if src_file.exists():
            dst_file = src_dir / file_name
            try:
                shutil.move(str(src_file), str(dst_file))
                print(f"✓ 移動: {file_name} -> src/{file_name}")
            except Exception as e:
                print(f"✗ エラー ({file_name}): {e}")
    
    # データファイルを移動
    data_files = [
        "2010-all.csv",
        "world-all.csv",
        "depth.csv"
    ]
    
    for file_name in data_files:
        src_file = base_dir / file_name
        if src_file.exists():
            dst_file = data_dir / file_name
            try:
                shutil.move(str(src_file), str(dst_file))
                print(f"✓ 移動: {file_name} -> data/{file_name}")
            except Exception as e:
                print(f"✗ エラー ({file_name}): {e}")
    
    # デバッグファイルを移動
    debug_files = list(base_dir.glob("debug_html_*.html"))
    for debug_file in debug_files:
        dst_file = debug_dir / debug_file.name
        try:
            shutil.move(str(debug_file), str(dst_file))
            print(f"✓ 移動: {debug_file.name} -> debug/{debug_file.name}")
        except Exception as e:
            print(f"✗ エラー ({debug_file.name}): {e}")
    
    print("\n整理が完了しました！")
    
    # パス設定を更新
    print("\nパス設定を更新中...")
    import subprocess
    import sys
    update_script = base_dir / "update_paths.py"
    if update_script.exists():
        try:
            subprocess.run([sys.executable, str(update_script)], check=False)
        except Exception as e:
            print(f"パス更新スクリプトの実行エラー: {e}")

if __name__ == "__main__":
    organize_files()





















