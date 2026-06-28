"""
YOLO ML Backend 启动脚本 (Python 版)

等价于 start.bat，更适合在终端中调试和查看日志。
按顺序执行:
  1. 读取配置 (config.json → 环境变量)
  2. 清理 Label Studio 缓存
  3. 模型检查
  4. 启动 Flask 服务器

用法:
  python start.py                   # 正常启动
  python start.py --port 9091       # 自定义端口
  python start.py --no-cache        # 跳过缓存清理
  python start.py --check-only      # 只做模型检查，不启动服务器
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path

logger = logging.getLogger("start")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_handler)

# 脚本所在目录
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent

# 默认配置路径
CONFIG_EXAMPLE = SCRIPT_DIR / "config.example.json"
CONFIG_FILE = SCRIPT_DIR / "config.json"
WSGI_SCRIPT = SCRIPT_DIR / "_wsgi.py"


def load_config() -> dict:
    """读取配置: 先从 config.example.json 加载, 再用 config.json 覆盖."""
    config = {}

    # 1) 模板
    if CONFIG_EXAMPLE.exists():
        with open(CONFIG_EXAMPLE, encoding="utf-8") as f:
            config.update(json.load(f))
        logger.info(f"已加载模板配置: {CONFIG_EXAMPLE.name}")
    else:
        logger.warning(f"未找到模板配置: {CONFIG_EXAMPLE}")

    # 2) 用户配置覆盖
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, encoding="utf-8") as f:
            config.update(json.load(f))
        logger.info(f"已加载用户配置: {CONFIG_FILE.name}")
    else:
        logger.warning(f"未找到用户配置: {CONFIG_FILE}")

    return config


def set_env_vars(config: dict):
    """将配置项设为环境变量 (只设置非空值)."""
    for key, value in config.items():
        if value is not None and value != "":
            os.environ[key] = str(value)
            # 对 API Key 做脱敏打印
            if "API_KEY" in key or "TOKEN" in key:
                display = str(value)[:8] + "..." if len(str(value)) > 8 else "***"
                logger.info(f"  环境变量 {key} = {display}")
            else:
                logger.info(f"  环境变量 {key} = {value}")


def clean_cache():
    """清理 Label Studio 的本地图片缓存."""
    cache_dir = Path(os.environ.get("LOCALAPPDATA", "~")) / "label-studio" / "Cache"
    cache_dir = cache_dir.expanduser()

    if not cache_dir.exists():
        logger.info(f"  缓存目录不存在, 跳过: {cache_dir}")
        return

    try:
        count = 0
        for f in cache_dir.iterdir():
            if f.is_file():
                f.unlink()
                count += 1
        logger.info(f"  已清理 {count} 个缓存文件: {cache_dir}")
    except Exception as e:
        logger.warning(f"  缓存清理失败 (不影响启动): {e}")


def check_model() -> bool:
    """直接加载 YOLO 模型，验证模型文件可用."""
    logger.info("运行模型检查...")
    try:
        # 将 src_yls 加入 sys.path 以便 import model
        sys.path.insert(0, str(SCRIPT_DIR))
        from ultralytics import YOLO as UltralyticsYOLO
        model_path = str(SCRIPT_DIR / "models" / "m960n.pt")
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return False
        model = UltralyticsYOLO(model_path)
        names = model.names
        logger.info(f"  模型: {os.path.basename(model_path)}")
        logger.info(f"  类别 ({len(names)}): {list(names.values())}")
        del model
        logger.info("模型检查通过 ✓")
        return True
    except Exception as e:
        logger.error(f"模型检查失败: {e}")
        return False


def start_server(wsgi_args: list):
    """启动 Flask 服务器 (替换当前进程)."""
    cmd = [sys.executable, str(WSGI_SCRIPT)] + wsgi_args
    logger.info(f"启动服务器: {' '.join(cmd)}")
    logger.info("=" * 50)
    # 用 exec 替换当前进程, 让服务器直接接管 stdout
    os.chdir(str(SCRIPT_DIR))
    os.execv(sys.executable, cmd)


def main():
    parser = argparse.ArgumentParser(description="YOLO ML Backend 启动脚本")
    parser.add_argument("--port", type=int, default=9090, help="服务器端口 (默认: 9090)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--debug", action="store_true", help="Flask 调试模式")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--no-cache", action="store_true", help="跳过缓存清理")
    parser.add_argument("--check-only", action="store_true", help="只做模型检查, 不启动服务器")
    parser.add_argument("--model-dir", help="模型文件目录")
    args = parser.parse_args()

    print()
    logger.info("=" * 50)
    logger.info("YOLO ML Backend for Label Studio")
    logger.info("=" * 50)
    print()

    # ---- Step 1: 加载配置 ----
    logger.info("[Step 1/4] 加载配置...")
    config = load_config()
    set_env_vars(config)

    # ---- Step 2: 清理缓存 ----
    logger.info("[Step 2/4] 清理 Label Studio 缓存...")
    if not args.no_cache:
        clean_cache()
    else:
        logger.info("  已跳过 (--no-cache)")

    # ---- Step 3: 模型检查 ----
    logger.info("[Step 3/4] 模型检查...")
    wsgi_args = []
    if args.port:
        wsgi_args += ["--port", str(args.port)]
    if args.host:
        wsgi_args += ["--host", args.host]
    if args.log_level:
        wsgi_args += ["--log-level", args.log_level]
    if args.model_dir:
        wsgi_args += ["--model-dir", args.model_dir]
    if args.debug:
        wsgi_args += ["--debug"]

    if not check_model():
        sys.exit(1)

    # ---- Step 4: 启动服务器 ----
    if args.check_only:
        logger.info("[Step 4/4] 跳过 (--check-only)")
        logger.info("模型检查完成 ✓")
        return

    logger.info("[Step 4/4] 启动服务器...")
    start_server(wsgi_args)


if __name__ == "__main__":
    main()
