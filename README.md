# Basilisk Attitude Project

这是一个基于 AVS Lab Basilisk 的航天器姿态仿真项目骨架。

当前阶段目标：
- 以 `scenario_AttGuidance` 为基线，保留 6-DOF、hill pointing、reaction wheel pyramid 的主链路理解
- 明确项目 requirements
- 先把仓库结构搭好，再逐步放入自定义场景、控制逻辑和测试

## 当前状态

- 官方基线场景 `scenario_AttGuidance` 已在单独的 Basilisk 环境中跑通
- 主链路已经梳理清楚：
  `modeRequest -> hillPoint -> trackingError -> mrpFeedbackRWs -> rwMotorTorque -> rwStateEffector`
- 本仓库用于承载后续自己的项目代码与文档

## 建议工作流

1. 在 WSL 中激活已经配置好的 Basilisk Python 环境
2. 在本仓库中编写自己的场景、配置和测试
3. 将成熟的控制链路从官方示例逐步迁移为自己的模块化实现

参考命令：

```bash
cd /home/llizhi/avslab/basilisk-develop
source .venv-linux/bin/activate
cd /mnt/e/WSL/basilisk-attitude-project
```

## 目录结构

```text
basilisk-attitude-project/
├─ docs/                文档与需求说明
├─ configs/             任务模式、仿真参数、执行机构参数
├─ scenarios/           自定义场景脚本
├─ scripts/             运行与辅助脚本
├─ src/
│  ├─ actuators/        执行机构相关代码
│  ├─ modes/            任务模式与模式管理
│  ├─ sensors/          传感器建模与接口
│  └─ simulation/       仿真装配与主流程
├─ tests/               单元测试与集成测试
└─ results/             本地输出结果与图像
```

## 第一阶段建议交付物

- `docs/requirements.md`
- 一个基于官方示例改造的自定义场景脚本
- 至少一组可以复现实验结果的配置文件
- 一份能够说明姿态误差和控制力矩变化趋势的输出图

## 基线场景

当前仓库已经提供一个项目内的基线场景入口：

- `scenarios/scenario_hill_point_baseline.py`
- `configs/baseline.yaml`
- `scripts/run_baseline.sh`

建议使用方式：

```bash
cd /mnt/e/WSL/basilisk-attitude-project
bash scripts/run_baseline.sh
```

如果想显式指定配置文件：

```bash
cd /mnt/e/WSL/basilisk-attitude-project
source /home/llizhi/avslab/basilisk-develop/.venv-linux/bin/activate
python3 scenarios/scenario_hill_point_baseline.py --config configs/baseline.yaml
```
