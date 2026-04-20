"""Utility helpers for the project-local pure Python truth model."""
# 模块文档字符串：为项目本地的纯 Python 真实模型提供通用工具函数的统一入口

from src.utils.frames import (
    mrp_derivative,
    mrp_to_dcm,
    orbital_frame_dcm,
    rotate_body_to_inertial,
    rotate_inertial_to_body,
    switch_to_shadow_mrp,
)
# 从参考系变换子模块导入以下函数：
# - mrp_derivative：计算 MRP 姿态参数的时间导数
# - mrp_to_dcm：将 MRP 转换为方向余弦矩阵（DCM）
# - orbital_frame_dcm：计算轨道坐标系相对于惯性系的 DCM
# - rotate_body_to_inertial：将体坐标系向量旋转到惯性系
# - rotate_inertial_to_body：将惯性系向量旋转到体坐标系
# - switch_to_shadow_mrp：将 MRP 映射到影子集以避免奇异

from src.utils.math_utils import rk4_step, safe_normalize, skew_symmetric
# 从数值计算子模块导入以下函数：
# - rk4_step：四阶龙格-库塔法单步积分
# - safe_normalize：安全向量归一化（防除零）
# - skew_symmetric：构造向量的反对称矩阵

__all__ = [
    "mrp_derivative",
    "mrp_to_dcm",
    "orbital_frame_dcm",
    "rk4_step",
    "rotate_body_to_inertial",
    "rotate_inertial_to_body",
    "safe_normalize",
    "skew_symmetric",
    "switch_to_shadow_mrp",
]
# 定义模块的公开接口列表。
# 当外部使用 `from src.utils import *` 时，仅会导入 `__all__` 中列出的名称。
# 这样做的好处是：
# 1. 将分散在 `frames` 和 `math_utils` 中的函数聚合到一个统一的命名空间下
# 2. 外部调用者无需关心这些函数具体来自哪个子模块，只需 `from src.utils import ...` 即可
# 3. 明确告知使用者本模块对外提供的核心功能