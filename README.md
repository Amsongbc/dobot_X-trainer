## 推理入口脚本 (experiments/)

本仓库提供 5 个推理入口，按「模型在哪里跑」和「是否支持人工介入」区分：

| 脚本 | 模型位置 | 后端/协议 | 人工介入 | 说明 |
| --- | --- | --- | --- | --- |
| `run_inference.py` | 本地 checkpoint | `Imitate_Model`（ModelTrain） | ✗ | 原始本地推理示例 |
| `run_inference_local_with_intervention.py` | 本地 checkpoint | `Imitate_Model` | ✓ | 本地推理 + 主手按键接管 |
| `run_ws_inference.py` | 远程服务器 | Motus，JSON + base64 | ✗ | WebSocket 远程推理 |
| `run_ws_inference_openpi.py` | 远程服务器 | OpenPI (PI-0.5)，msgpack | ✗ | 官方 openpi-client 远程推理 |
| `run_ws_inference_with_intervention.py` | 远程服务器 | OpenPI (PI-0.5)，msgpack | ✓ | 远程推理 + 主手按键接管 |

**远程推理（`run_ws_*`）公共功能**：动作块（action chunk）截断执行、时序集成
（temporal ensemble，前后 chunk 重叠段指数加权融合）、三路 RealSense 相机线程采图、
相机画面视频录制、`--dry-run`（只推理不动真机）、`--check-only`（只测服务器连通性）。
服务端部署与 SSH 隧道配置见 [dobot_client/README.md](dobot_client/README.md)。

**人工介入（intervention）**：推理执行过程中，按住任一主手的录制键即可实时接管对应从手臂
（左右臂独立），主手位移以增量映射到从手；松开按键后该臂交还 policy 控制。
远程版在交还时会丢弃介入前推理出的剩余动作块，用当前真机状态重新请求推理，
并带有单步步长限幅与关节/笛卡尔工作空间安全边界检查（越界立即停机并亮红灯）。
该机制可用于演示纠正数据（correction data）的采集与人机共享控制研究。
