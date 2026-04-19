#!/usr/bin/env python3
"""CLI entrypoint for the project-local hill-pointing baseline scenario."""
# 模块文档字符串：说明本文件是项目本地Hill只想基准场景的命令入口

from __future__ import annotations
# 启用 postponed evaluation of annotations（推迟注解求值），允许使用尚未定义的类型名


import argparse # 导入标准库 argparse，用于解析命令行参数
import sys #导入sys模块，用于操作系统相关功能（如修改模块搜索路径、退出程序）
from pathlib import Path  # 从 pathlib 导入 Path 类，提供面向对象的文件系统路径操作
from typing import Any # 从 typing 导入 Any 类型，表示任意类型

import yaml  # 导入第三方库 PyYAML，用于解析 YAML 配置文件


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "baseline.yaml"
# 计算仓库根目录的绝对路径：当前文件的绝对路径 -> 上一级目录 -> 再上一级目录（因为本文件在 src/cli/ 下，根目录在两层之上）
# 默认配置文件路径：仓库根目录下的 configs/baseline.yaml

# 如果仓库根目录的字符串形式不在 Python 模块搜索路径中，则将其插入到路径列表的最前面，以确保可以导入 src 包
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 从项目的仿真模块中导入主场景运行函数 run_scenario
from src.simulation.hill_point_baseline import run_scenario

# 定义一个函数，接受一个 Path 对象，返回字典（键为字符串，值为任意类型）
def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""# 函数文档：加载 YAML 配置文件
    with path.open("r", encoding="utf-8") as handle:
        # 使用 UTF-8 编码以只读模式打开文件
        data = yaml.safe_load(handle) or {}
        # 使用 safe_load 安全加载 YAML 内容，如果为空则返回空字典
    if not isinstance(data, dict):
        # 如果加载后的数据不是字典类型
        raise ValueError(f"Expected a mapping in {path}, but got {type(data).__name__}.")
        # 抛出异常，提示期望一个映射（字典）但实际得到其他类型
    return data

# 定义命令行参数解析函数，返回解析结果命名空间
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the project-local Basilisk hill-pointing baseline.")
    # 创建参数解析器，设置描述信息
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    # 添加 --config 参数，类型为 Path，默认值为之前定义的配置文件路径，帮助信息
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Override the config and display plots interactively.",
    )
    # 添加 --show-plots 开关，若指定则在运行后显示绘图（覆盖配置文件中的设置）
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Override the config and skip saving plot images.",
    )
    # 添加 --no-save 开关，若指定则不保存绘图图像（覆盖配置文件中的设置）
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config) # 加载指定路径的 YAML 配置文件

    if args.show_plots:
        config["output"]["show_plots"] = True
    if args.no_save:
        config["output"]["save_plots"] = False

    saved_paths = run_scenario(config) # 调用核心仿真函数，传入配置字典，返回保存的文件路径列表

    if saved_paths:
        print("Saved baseline outputs:")
        for path in saved_paths:
            print(f"  - {path}")
    else:
        print("Scenario completed without saving plots.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
