## 推理入口脚本 (experiments/)

本仓库提供 5 个推理入口，按「模型在哪里跑」和「是否支持人工介入」区分：

| 脚本 | 模型位置 | 后端/协议 | 人工介入 | 说明 |
| --- | --- | --- | --- | --- |
| `run_inference.py` | 本地 checkpoint | `Imitate_Model`（ModelTrain） | ✗ | 原始本地推理示例 |
| `run_inference_local_with_intervention.py` | 本地 checkpoint | `Imitate_Model` | ✓ | 本地推理 + 主手按键接管 |
| `run_ws_inference.py` | 远程服务器 | Motus，JSON + base64 | ✗ | WebSocket 远程推理 |
| `run_ws_inference_openpi.py` | 远程服务器 | OpenPI (PI-0.5)，msgpack | ✗ | 官方 openpi-client 远程推理 |
| `run_ws_inference_openpi_with_intervention.py` | 远程服务器 | OpenPI (PI-0.5)，msgpack | ✓ | 远程推理 + 主手按键接管 |

**远程推理（`run_ws_*`）公共功能**：动作块（action chunk）截断执行、时序集成
（temporal ensemble，前后 chunk 重叠段指数加权融合）、三路 RealSense 相机线程采图、
相机画面视频录制、`--dry-run`（只推理不动真机）、`--check-only`（只测服务器连通性）。
服务端部署与 SSH 隧道配置见 [dobot_client/README.md](dobot_client/README.md)。

**人工介入（intervention）**：推理执行过程中，按住任一主手的录制键即可实时接管对应从手臂
（左右臂独立），主手位移以增量映射到从手；松开按键后该臂交还 policy 控制。
远程版在交还时会丢弃介入前推理出的剩余动作块，用当前真机状态重新请求推理，
并带有单步步长限幅与关节/笛卡尔工作空间安全边界检查（越界立即停机并亮红灯）。
该机制可用于演示纠正数据（correction data）的采集与人机共享控制研究。

---

## 在线 RLT 训练入口 (experiments/run_stage2_env_client.py)

真机侧客户端，把机械臂封装成远程 environment 交给训练端驱动。单文件，不依赖
`experiments/` 下其他模块。

### 启动

```bash
python experiments/run_stage2_env_client.py \
  --server-host 127.0.0.1 \
  --server-port 18000
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--server-host` / `--server-port` | `127.0.0.1` / `18000` | 训练端地址 |
| `--robot-port` / `--hostname` | `6001` / `127.0.0.1` | `robot_node` 地址 |
| `--instruction` | `pour water` | 默认 prompt |
| `--control-hz` | `10.0` | 控制频率 |
| `--always-actor` | `False` | 全程用 actor |
| `--crop-top-camera` | `False` | 顶部相机裁剪 |
| `--dry-run` | `False` | 不驱动真机 |
| `--reconnect-delay` | `3.0` | 断线重连间隔（秒） |

需要：`robot_node` 已运行、三路 RealSense 已连接、在真 TTY 下运行（建议 tmux）。
跨机时在真机侧开隧道 `ssh -N -L 18000:127.0.0.1:18000 user@gpu-host`。

### 传输

客户端主动连接训练端。每帧 = 8 字节大端长度头（`!Q`）+
`openpi_client.msgpack_numpy` 包体。

### server 发给 client

```python
{"command": "reset", "prompt": str}
{"command": "step",  "actions": np.ndarray}   # [C, 14] float
```

动作 14 维：`[左臂 6 关节(rad), 左夹爪(0~1), 右臂 6 关节(rad), 右夹爪(0~1)]`，绝对值。

### client 返回给 server

`reset`：

```python
{"ok": True, "observation": observation}
```

`step`：

```python
{
  "ok": True,
  "observation": observation,
  "executed_actions": np.ndarray,   # [C, 14] float64，真机实际执行的动作
  "rewards": np.ndarray,            # [C] float32
  "done": bool,
  "info": {...},
}
```

出错：

```python
{"ok": False, "error": str}
```

`observation`：

```python
{
  "state": np.ndarray,              # [14] float32
  "images": {
    "cam_high":        np.ndarray,  # [3, H, W] uint8
    "cam_left_wrist":  np.ndarray,
    "cam_right_wrist": np.ndarray,
  },
  "prompt": str,
}
```

`info`：

| 键 | 类型 | 含义 |
| --- | --- | --- |
| `steps_executed` | int | 实际执行步数，其余为补齐 |
| `intervention_mask` | `[C,2]` bool | 逐步、逐臂：该步是否由人操作 |
| `intervention_occurred` | bool | 本次是否有人工介入 |
| `stale_policy_discarded` | bool | 本次下发的动作块被整块丢弃 |
| `action_chunk_truncated` | bool | 恒为 `False` |
| `actor_switch_requested` | bool | 请求下一 chunk 切换到 actor |
| `discard_episode` | bool | 本回合不要写入 replay |
| `anchor_marked` | bool | 本次标记了倒车 anchor |
| `success` | bool | 仅在回合结束时出现 |
| `robot_fault` | bool | 仅故障时出现 |
| `discard_transition` | bool | 仅故障时出现，本条 transition 丢弃 |
| `fault_message` | str | 仅故障时出现 |

### 按键

回合进行中（键盘）：

| 按键 | 作用 |
| --- | --- |
| `s` / 空格 | 成功，reward +1，结束回合 |
| `f` | 失败，结束回合 |
| `p` | 进展，reward +0.5，不结束回合 |
| `r` | 丢弃本回合 |
| `b` | 下一 chunk 切换到 actor |
| `a` | 把当前这一步记为倒车 anchor |
| `Ctrl+C` | 停止客户端 |

主手（硬件）：

| 操作 | 作用 |
| --- | --- |
| 按住录制键 | 接管对应从臂，松开交还 policy（左右臂独立） |
| 短按 A 键 | 解锁/锁定该侧主臂扭矩 |

标记过 anchor 后，回合结束时会询问：直接回车 = 机械臂回到 anchor 并开始新回合，
输入 `n` = 正常复位。anchor 保留，可反复倒车。
