"""Bootstrap helpers for loading the official Basilisk BSK_Sim modules."""
# 模块文档字符串：提供加载官方 Basilisk BSK_Sim 模块的引导辅助函数

from __future__ import annotations

import os
import sys
from pathlib import Path


# 计算项目仓库根目录的绝对路径：
# __file__ 是当前脚本的路径（例如 src/bootstrap/bsk_sim_loader.py）
# .resolve() 获取绝对路径
# .parents[2] 向上回退两级目录（bootstrap -> src -> 仓库根目录）
REPO_ROOT = Path(__file__).resolve().parents[2]

# 定义一个函数，接受候选路径，尝试定位 BSK_Sim 的真正根目录，若找不到返回 None
def normalize_bsk_sim_candidate(path: Path) -> Path | None:
    """Accept either a BSK_Sim root or a Basilisk repo root."""
    # 依次检查两种可能的目录结构：
    # 1. 传入的路径本身
    # 2. 传入路径下的 examples/BskSim 子目录（对应官方 Basilisk 仓库的标准布局）
    for candidate in (path, path / "examples" / "BskSim"):
        if (candidate / "BSK_masters.py").exists():
            return candidate.resolve()
    return None

# 定义函数，定位并返回官方 Basilisk BSK_Sim 示例的根目录路径
def resolve_bsk_sim_root() -> Path:
    """Find the official Basilisk BSK_Sim example root used as the baseline."""
    candidates = []

    env_path = os.environ.get("BASILISK_BSKSIM_ROOT")
    if env_path:
        candidates.append(Path(env_path))

    # 添加与项目仓库同级的 basilisk-develop 目录（假设官方仓库克隆在项目父目录下）
    candidates.append(REPO_ROOT.parent / "basilisk-develop")

    # 添加用户家目录下的标准安装路径（常见于 AVSLab 团队教程中的位置）
    candidates.append(Path.home() / "avslab" / "basilisk-develop" / "examples" / "BskSim")

    for candidate in candidates:
        normalized = normalize_bsk_sim_candidate(candidate)
        if normalized is not None:
            return normalized

    raise FileNotFoundError(
        "Could not find the Basilisk BSK_Sim example root. "
        "Set BASILISK_BSKSIM_ROOT, keep a sibling basilisk-develop checkout, "
        "or install Basilisk under ~/avslab/basilisk-develop."
    )


def bootstrap_bsk_paths() -> Path:
    """Add the official BSK_Sim example directories to ``sys.path``."""
    bsk_sim_root = resolve_bsk_sim_root()
    extra_paths = [
        str(bsk_sim_root),
        str(bsk_sim_root / "plotting"),
    ]

    # 这些目录必须先进入 sys.path，
    # 后面的 BSK_masters / BSK_Dynamics / BSK_Fsw 才能正常导入。
    for path in reversed(extra_paths):
        if path not in sys.path:
            sys.path.insert(0, path)

    return bsk_sim_root

# 在模块加载时立即执行路径引导，并将得到的 BSK_Sim 根目录存入常量 BSK_SIM_ROOT
# 这样后续导入代码可以依赖此常量获取路径
BSK_SIM_ROOT = bootstrap_bsk_paths()

# 这里集中导入官方 BSK_Sim 模块，
# 这样项目里的场景文件就不需要重复处理路径和 import 细节。

# 从 BSK_masters 模块导入核心类 BSKScenario 和 BSKSim
# noqa: E402 告诉 flake8 忽略此行违反的 PEP8 E402 规则（模块级导入不在文件顶部）
from BSK_masters import BSKScenario, BSKSim  # noqa: E402
import BSK_Dynamics  # noqa: E402
import BSK_Fsw  # noqa: E402

# 导入绘图模块并设置别名 BSK_plt，便于后续调用
import BSK_Plotting as BSK_plt  # noqa: E402
