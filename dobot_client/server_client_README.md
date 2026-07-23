# dobot_client — 远程推理客户端与服务端部署

本目录用于"**真机在本地、模型在远程 GPU 服务器**"的部署方式：真机侧采集三路相机图像与
关节状态，通过 SSH 隧道 + WebSocket 发送给远程推理服务器，接收动作块并执行。

支持两种推理后端：

| 后端 | 服务端启动 | 协议 | 真机客户端（experiments/） |
| --- | --- | --- | --- |
| **OpenPI π0.5** | `serve_openpi05_server.py` | msgpack（官方 `openpi-client`） | `run_ws_inference_openpi.py`、`run_ws_inference_openpi_with_intervention.py` |
| **Motus** | `inference.sh` → `server_vlm_mask.py` | JSON + base64 JPEG | `run_ws_inference.py` |

> 两种协议互不兼容，客户端脚本必须与对应后端配套使用。

## 目录内容

| 文件 | 运行位置 | 后端 | 用途 |
| --- | --- | --- | --- |
| `serve_openpi05_server.py` | GPU 服务器 | OpenPI | 启动 π0.5 Dobot 策略服务器（包装 OpenPI 官方 `scripts/serve_policy.py`） |
| `server_vlm_mask.py` | GPU 服务器 | Motus | Motus 推理服务端（FastAPI，WebSocket `/ws`），需放入 Motus 工程内运行 |
| `inference.sh` | GPU 服务器 | Motus | 启动 `server_vlm_mask.py`，`TASK=cook/pour` 选择任务配置 |
| `client_ws.py` | 本机 | Motus | **独立测试客户端**（不接真机）：健康检查 / mock / 单帧或循环推理，图像从磁盘读取 |
| `test_frames/` | 本机 | 通用 | 测试用三路相机帧（top / left / right）与状态 CSV |

> 注意：`server_vlm_mask.py` 和 `inference.sh` 依赖 Motus 工程代码
> （`models.motus_wan_vlm_direct_mask` 等），需拷贝到服务器上 Motus 工程的 `dobot/` 目录下运行；
> `serve_openpi05_server.py` 则依赖服务器上已克隆的 OpenPI 仓库。二者均不在本仓库内直接运行模型。

## 网络架构（两种后端相同）

```
真机客户端 ──► 127.0.0.1:18000 ──SSH 隧道──► GPU 服务器 <IP>:<PORT> ──► 推理服务
```

SSH 隧道（本机保持运行，无输出属正常）：

```bash
ssh -i <私钥> -p <端口> \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -N -L 127.0.0.1:18000:<目标IP>:<PORT> <用户>@<入口机>
```

若 SSH 入口机与 GPU 机不是同一台，`-L` 的目标须写 GPU 机的内网 IP，而非 `127.0.0.1`。

---

## 一、OpenPI π0.5 后端

### 1. 服务器端启动

前置条件：已克隆 [OpenPI](https://github.com/Physical-Intelligence/openpi) 仓库并装好其环境；
已准备 π0.5 Dobot checkpoint 目录（须包含 `model.safetensors` 和归一化统计 `assets/`）。

```bash
python dobot_client/serve_openpi05_server.py \
  --openpi-dir <openpi仓库路径> \
  --checkpoint-dir <checkpoint路径>/29999_pytorch \
  --port 8000
```

- 路径也可用环境变量 `OPENPI_DIR` / `OPENPI_CHECKPOINT_DIR` 提供；
- 训练配置默认 `pi05_dobot_full`，可用 `--config` 覆盖；
- `--default-prompt "<指令>"` 设置客户端未传指令时的默认 prompt；
- `--record` 用 OpenPI 的 PolicyRecorder 记录请求/响应；
- `--dry-run` 只打印将要执行的命令，不真正启动。

### 2. 本机客户端

```bash
# 1) 连通性检查（HTTP /healthz + WebSocket 握手，不动真机）
python experiments/run_ws_inference_openpi.py \
  --ws-host 127.0.0.1 --ws-port 18000 --check-only

# 2) dry run（真实推理，但不下发机械臂指令）
python experiments/run_ws_inference_openpi.py \
  --ws-host 127.0.0.1 --ws-port 18000 \
  --robot-port 6001 --hostname 127.0.0.1 \
  --instruction "pour water" --action-chunk-len 1 --dry-run

# 3) 真机运行
python experiments/run_ws_inference_openpi.py \
  --ws-host 127.0.0.1 --ws-port 18000 \
  --robot-port 6001 --hostname 127.0.0.1 \
  --instruction "pour water"

# 4) 真机运行 + 人工介入（按住主手录制键接管对应从手臂，松开交还 policy）
python experiments/run_ws_inference_with_intervention.py \
  --ws-host 127.0.0.1 --ws-port 18000 \
  --robot-port 6001 --hostname 127.0.0.1 \
  --instruction "pour water"
```

### 协议（openpi-client，msgpack）

observation：`state`（14 维关节状态，左臂 7 + 右臂 7，含夹爪）、
`images.cam_high / cam_left_wrist / cam_right_wrist`（CHW、uint8 RGB）、`prompt`（指令文本）。
返回 `actions`：形状 `[T, 14]` 的动作块。

---

## 二、Motus 后端

### 1. 服务器端启动

```bash
cd <Motus工程根目录>
export PYTHONPATH="<Motus工程根目录>:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0 CKPT_DIR=<checkpoint路径> TASK=cook \
HOST=0.0.0.0 PORT=8000 bash dobot/inference.sh
```

### 2. 本机逐级验证（`client_ws.py`，不接真机）

```bash
pip install websocket-client numpy pillow            # 首次
curl http://127.0.0.1:18000/health                    # 隧道是否通
python dobot_client/client_ws.py 127.0.0.1 --port 18000 --check-only   # WS 健康检查
python dobot_client/client_ws.py 127.0.0.1 --port 18000 --mock         # mock（不跑模型）

# 真实推理（磁盘图像，三相机分开传）
python dobot_client/client_ws.py 127.0.0.1 --port 18000 \
  --top_image dobot_client/test_frames/top.jpg \
  --left_wrist_image dobot_client/test_frames/left.jpg \
  --right_wrist_image dobot_client/test_frames/right.jpg \
  --state_csv "0,0,0,0,0,0,0,0,0,0,0,0,0,0"

# 10Hz 循环推理
python dobot_client/client_ws.py 127.0.0.1 --port 18000 --hz 10 --count 5 \
  --image <拼接图路径> --state_csv "0,0,0,0,0,0,0,0,0,0,0,0,0,0"
```

### 3. 真机运行

```bash
python experiments/run_ws_inference.py \
  --ws-host 127.0.0.1 --ws-port 18000 \
  --robot-port 6001 --hostname 127.0.0.1
```

### 协议（JSON + base64）

请求：`type`（`health` / `mock` / `inference`）、`images`（base64 JPEG 列表）、
`proprio_data`（14 维状态）、可选 `instruction`。响应：`predicted_actions`（`[T, 14]` 动作块）。

---

## 真机客户端通用参数

`--action-chunk-len` 截断每次执行的动作块长度、`--temporal-ensemble` 开启前后动作块重叠融合、
`--record-video` / `--video-dir` 录制相机画面、`--crop-top-camera` 裁剪顶部相机视野、
`--dry-run` 只推理不动真机。

## 排错

```bash
# 确认 SSH 实际落在哪台机器
ssh -i <私钥> -p <端口> <用户>@<入口机> "hostname"
# 从入口机直接访问 GPU 机的服务（OpenPI 用 /healthz，Motus 用 /health）
ssh -i <私钥> -p <端口> <用户>@<入口机> "curl -s http://<GPU机内网IP>:<PORT>/healthz"
```
