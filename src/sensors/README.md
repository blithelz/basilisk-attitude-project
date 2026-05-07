# Sensors

这里用于放置传感器接口和简化测量模型。

当前目录同时保留两条线：

- `simple_nav.py`
  这是前面 Basilisk 工程骨架里为了快速跑通场景保留的兼容层。

- `gyro.py` / `magnetometer.py` / `sun_sensor.py` / `sensor_model.py`
  这是第 3 周新增的纯 Python 传感器层。
  它们直接消费 `src/truth/` 输出的真值历史，
  再叠加采样率、偏置、噪声和可见性逻辑，生成后续估计器要使用的“测量值”。

这一层的目标不是替代 Basilisk，
而是先把 `truth -> measurement` 的接口单独建清楚，
后面再把同样的测量接口接回 Basilisk 集成仿真。
