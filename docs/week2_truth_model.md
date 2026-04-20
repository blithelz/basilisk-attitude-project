# 第二周：真实模型

## 目标

第二周为近地轨道（LEO）小卫星引入项目本地的真实模型层。

实现严格遵循以下顺序：

1. 刚体姿态动力学
2. 轨道递推
3. 环境真值生成
4. 外部干扰力矩建模

关键的分层原则是：

- 状态递推与环境真值生成分离
- 环境真值生成与干扰力矩建模分离

这样做可以保持接口清晰，为后续几周做准备：

- 第三周的传感器可以从 `environment/` 读取真值
- 第五周的估计器可以从 `sensors/` 读取测量值
- 控制器无需关心真值数据的来源

## 代码布局

第二周新增的模块位于：

- `src/truth/rigid_body.py`          # 刚体姿态动力学真值
- `src/truth/orbit.py`               # 轨道递推真值
- `src/environment/leo.py`           # LEO 环境真值生成
- `src/disturbances/torques.py`      # 外部干扰力矩建模

基准场景按以下顺序组装这些模块：

- 配置真实模型
- 挂接真值/环境数据记录器
- 运行官方 `BSK_Dynamics + BSK_Fsw` 仿真
- 后处理环境真值数据
- 后处理干扰力矩数据

## 当前范围

目前的干扰模型输出以下真值力矩时间历程：

- 重力梯度力矩
- 简化的气动阻力矩
- 简化的太阳光压力矩
- 残余磁偶极矩

现阶段这些力矩仅作为干净的真值/输出层存在。
配置字段 `disturbances.apply_to_dynamics` 被保留，用于未来决定是否将建模力矩重新注入动力学回路。

## 配置文件

第二周的基准配置文件为：

- `configs/leo_truth.yaml`

它新增了以下显式配置节：

- `spacecraft`        # 航天器物理参数
- `truth_model`       # 真实模型设置
- `environment`       # 环境模型设置
- `disturbances`      # 干扰力矩设置

## 运行命令

进入项目目录并激活虚拟环境：

```bash
cd /mnt/e/WSL/basilisk-attitude-project
source /home/llizhi/avslab/basilisk-develop/.venv-linux/bin/activate
python3 -m unittest discover -s tests -v


# Week 2 Truth Model

## Goal

Week 2 introduces a project-local truth-model layer for a LEO small satellite.

The implementation follows one strict order:

1. Rigid-body attitude dynamics
2. Orbit propagation
3. Environment truth generation
4. External disturbance torque modeling

The key separation rule is:

- state propagation is separate from environment generation
- environment generation is separate from disturbance modeling

This keeps the interfaces clean for later weeks:

- week 3 sensors can read truth values from `environment/`
- week 5 estimators can read measurements from `sensors/`
- controllers do not need to know where truth values came from

## Code Layout

The new week-2 modules live in:

- `src/truth/rigid_body.py`
- `src/truth/orbit.py`
- `src/environment/leo.py`
- `src/disturbances/torques.py`

The baseline scenario now wires them in this order:

- configure truth model
- attach truth/environment recorders
- run official `BSK_Dynamics + BSK_Fsw`
- post-process environment truth
- post-process disturbance torques

## Current Scope

The current disturbance model outputs truth torque histories for:

- gravity-gradient torque
- simplified aerodynamic-drag torque
- simplified solar-radiation-pressure torque
- residual magnetic-dipole torque

For now these torques are kept as a clean truth/output layer.
The configuration field `disturbances.apply_to_dynamics` is reserved for a later step if we decide to inject modeled torques back into the dynamics loop.

## Config

The week-2 baseline config is:

- `configs/leo_truth.yaml`

It adds explicit sections for:

- `spacecraft`
- `truth_model`
- `environment`
- `disturbances`

## Run

```bash
cd /mnt/e/WSL/basilisk-attitude-project
source /home/llizhi/avslab/basilisk-develop/.venv-linux/bin/activate
python3 -m unittest discover -s tests -v
```

Run the week-2 LEO truth scenario:

```bash
cd /mnt/e/WSL/basilisk-attitude-project
source /home/llizhi/avslab/basilisk-develop/.venv-linux/bin/activate
MPLBACKEND=Agg python3 scenarios/scenario_hill_point_baseline.py --config configs/leo_truth.yaml
```

## Outputs

Generated results are written under:

- `results/scenario_leo_truth_baseline/`

New week-2 plots include:

- `leo_truth_baseline_truthAttitude.png`
- `leo_truth_baseline_truthOrbit.png`
- `leo_truth_baseline_truthEnvironment.png`
- `leo_truth_baseline_disturbanceTorque.png`
