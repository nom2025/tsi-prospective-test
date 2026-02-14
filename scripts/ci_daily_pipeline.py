"""
CI Daily Pipeline — TSI前向き検証パイプライン
=============================================
GitHub Actions用のオーケストレーター。
データ取得 → TSI計算 → 前向き検証評価 を順次実行する。

設計思想:
- データ取得の失敗は許容（既存データで続行）
- TSI計算の失敗はパイプライン失敗
- 前向き検証評価の失敗は許容（情報表示のみ）
- 凍結されたスクリプトは一切変更しない
"""

import sys
import os
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "pipeline_log.txt"


def setup_logging():
    """コンソール + ファイルのデュアルログ設定"""
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # コンソール
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # ファイル
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def run_step(logger, name, script_path, max_retries=1, retry_delay=30, timeout=600):
    """
    Pythonスクリプトをサブプロセスとして実行する。

    Args:
        name: ステップ名
        script_path: スクリプトのパス
        max_retries: 最大リトライ回数
        retry_delay: リトライ間隔（秒）
        timeout: タイムアウト（秒）

    Returns:
        bool: 成功したかどうか
    """
    script = Path(script_path)
    if not script.exists():
        logger.error(f"[{name}] スクリプトが見つかりません: {script}")
        return False

    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.info(f"[{name}] リトライ {attempt}/{max_retries} ({retry_delay}秒待機)")
            time.sleep(retry_delay)

        logger.info(f"[{name}] 実行開始: {script.name}")
        start = time.time()

        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )

            elapsed = time.time() - start

            if result.returncode == 0:
                logger.info(f"[{name}] 成功 ({elapsed:.1f}秒)")
                if result.stdout.strip():
                    # 最後の数行だけログに出力
                    lines = result.stdout.strip().split("\n")
                    for line in lines[-10:]:
                        logger.info(f"  | {line}")
                return True
            else:
                logger.warning(f"[{name}] 失敗 (exit={result.returncode}, {elapsed:.1f}秒)")
                if result.stderr.strip():
                    for line in result.stderr.strip().split("\n")[-5:]:
                        logger.warning(f"  | {line}")

        except subprocess.TimeoutExpired:
            logger.warning(f"[{name}] タイムアウト ({timeout}秒)")
        except Exception as e:
            logger.warning(f"[{name}] 例外: {e}")

    return False


def main():
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("  TSI Daily Pipeline 開始")
    logger.info(f"  実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  プロジェクトルート: {PROJECT_ROOT}")
    logger.info(f"  CI環境: {os.environ.get('CI', 'false')}")
    logger.info("=" * 60)

    results = {}
    fetch_dir = PROJECT_ROOT / "データ自動取得" / "src"

    # ================================================================
    # 必要ディレクトリの事前作成（CI環境では存在しない場合がある）
    # ================================================================
    required_dirs = [
        PROJECT_ROOT / "生データ",
        PROJECT_ROOT / "データ分析結果出力",
        PROJECT_ROOT / "データ自動取得" / "data",
        PROJECT_ROOT / "データ自動取得" / "src",
    ]
    for d in required_dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("  必要ディレクトリを確認/作成完了")

    # ================================================================
    # Phase 1: データ取得（全て非クリティカル）
    # ================================================================
    logger.info("\n■ Phase 1: データ取得")

    #                       名前        スクリプト                              リトライ 待機  タイムアウト
    fetch_steps = [
        ("Japan EQ", fetch_dir / "fetch_earthquake_data.py",       1,     10,  300),
        ("World EQ", fetch_dir / "fetch_world_earthquake_data.py", 1,     10,  180),
        ("Deep EQ",  fetch_dir / "fetch_depth_data.py",            1,     10,  300),
        ("GNSS",     fetch_dir / "fetch_gnss_data.py",             1,     10,  300),
    ]
    # 注: GNSS取得はCI環境で遅くなりがち（国土地理院サイトのスクレイピング）
    # タイムアウト300秒×リトライ1回 = 最大310秒に抑制

    for name, script, retries, delay, timeout in fetch_steps:
        results[name] = run_step(logger, name, script, retries, delay, timeout)

    fetch_success = sum(1 for v in results.values() if v)
    logger.info(f"\n  データ取得結果: {fetch_success}/{len(fetch_steps)} 成功")

    # ================================================================
    # Phase 2: TSI計算（GNSSデータがあればクリティカル）
    # ================================================================
    logger.info("\n■ Phase 2: TSI計算")

    gnss_file = PROJECT_ROOT / "生データ" / "gnss_data.csv"
    if gnss_file.exists() and gnss_file.stat().st_size > 1000:
        results["TSI Calc"] = run_step(
            logger, "TSI Calc",
            PROJECT_ROOT / "tsi_daily_calc.py",
            max_retries=0, timeout=300
        )
    else:
        logger.warning("  GNSSデータなし → TSI計算スキップ")
        results["TSI Calc"] = False

    # ================================================================
    # Phase 3: 前向き検証評価（非クリティカル）
    # ================================================================
    logger.info("\n■ Phase 3: 前向き検証評価")

    tsi_frozen = PROJECT_ROOT / "データ分析結果出力" / "tsi_frozen.csv"
    if tsi_frozen.exists():
        results["Prospective Test"] = run_step(
            logger, "Prospective Test",
            PROJECT_ROOT / "prospective_test.py",
            max_retries=0, timeout=120
        )
    else:
        logger.info("  tsi_frozen.csv なし → 評価スキップ")
        results["Prospective Test"] = False

    # ================================================================
    # 結果サマリー
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("  パイプライン結果サマリー")
    logger.info("=" * 60)

    for step, success in results.items():
        status = "✓ 成功" if success else "✗ 失敗"
        logger.info(f"  {step:<20} {status}")

    # prospective_log.csv の状態確認
    log_file = PROJECT_ROOT / "prospective_log.csv"
    if log_file.exists():
        import pandas as pd
        try:
            log_df = pd.read_csv(log_file)
            logger.info(f"\n  prospective_log.csv: {len(log_df)}エントリ")
            if len(log_df) > 0:
                latest = log_df.iloc[-1]
                logger.info(f"  最新エントリ: {latest.get('date', 'N/A')}")
        except Exception:
            pass

    # 終了コード
    # TSI計算が失敗した場合のみパイプライン失敗とする
    # （データ取得の失敗は許容）
    if not results.get("TSI Calc", False) and gnss_file.exists() and gnss_file.stat().st_size > 1000:
        logger.error("\n  パイプライン失敗: TSI計算エラー")
        sys.exit(1)
    else:
        logger.info("\n  パイプライン完了")
        sys.exit(0)


if __name__ == "__main__":
    main()
