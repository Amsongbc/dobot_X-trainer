# Motus WebSocket 推理客户端使用说明

本文档说明如何通过 SSH 端口转发，在本机运行 `client_ws.py`，连接远程 GPU 服务器上的 Motus 推理服务（`dobot/server_vlm_mask.py`）。

## 架构概览

```
本机 client_ws.py  ──►  127.0.0.1:18000  ──SSH 隧道──►  远程 127.0.0.1:8000  ──►  server_vlm_mask.py (/ws)
```

| 位置       | 地址                  | 说明                         |
| ---------- | --------------------- | ---------------------------- |
| 远程服务器 | `127.0.0.1:8000`      | 推理服务实际监听端口         |
| SSH 隧道   | `18000 → 8000`        | 本机 18000 转发到远程 8000   |
| 本机客户端 | `127.0.0.1:18000`     | `client_ws.py` 连接地址      |


---

## 一、本机客户端

### 1. 私钥权限
chmod 600 /data2/liujingzhi/id_ed25519_5090

### 2. 安装依赖
pip install websocket-client numpy pillow

---

## 二、服务器端（远程 GPU 机器）
cd /data/liujingzhi/Motus
export PYTHONPATH="/data/liujingzhi/Motus/bak:/data/liujingzhi/Motus:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0 \
CKPT_DIR=/data/liujingzhi/Motus/checkpoint/dobot_c.pt \
TASK=cook \
HOST=0.0.0.0 \
PORT=8000 \
bash dobot/inference.sh

## 三、客户端（本机）
注意：SSH 登录的是 0004，服务在 0005（192.168.0.138），隧道必须转发到内网 IP，不能用 127.0.0.1。

需要 **两个终端**。

### 终端 1：建立 SSH 端口转发（保持运行、没有输出是正常的）
ssh -p 34135 -i /home/sbc/dobot_xtrainer/dobot_client/id_ed25519_5090 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -N -L 127.0.0.1:18000:192.168.0.138:8085 \
  root@116.63.180.90


### 终端 2：运行 WebSocket 客户端
ppip install websocket-client   # 首次需要
# 1) 确认隧道通了
curl http://127.0.0.1:18000/health
# 2) WebSocket 健康检查
python /home/sbc/dobot_xtrainer/dobot_client/client_ws.py 127.0.0.1 --port 18000 --check-only
# 3) mock 测试（不跑模型）
python /home/sbc/dobot_xtrainer/dobot_client/client_ws.py 127.0.0.1 --port 18000 --mock
# 4) 真实推理（单张拼接图）
python /home/sbc/dobot_xtrainer/dobot_client/client_ws.py 127.0.0.1 --port 18000 \
  --image /data2/liujingzhi/dobot_first_frame.jpg \
  --state_csv "0,0,0,0,0,0,0,0,0,0,0,0,0,0"
# 5) 真实推理（三相机分开传）
python /home/sbc/dobot_xtrainer/dobot_client/client_ws.py 127.0.0.1 --port 18000 \
  --top_image /data2/liujingzhi/top.jpg \
  --left_wrist_image /data2/liujingzhi/left.jpg \
  --right_wrist_image /data2/liujingzhi/right.jpg \
  --state_csv "0,0,0,0,0,0,0,0,0,0,0,0,0,0"
# 6) 10Hz 循环推理
python /data2/liujingzhi/client_ws.py 127.0.0.1 --port 18000 --hz 10 --count 5 \
  --image /data2/liujingzhi/dobot_first_frame.jpg \
  --state_csv "0,0,0,0,0,0,0,0,0,0,0,0,0,0"

python experiments/run_ws_inference.py \
  --ws-host 127.0.0.1 \
  --ws-port 18000 \
  --robot-port 6001 \
  --hostname 127.0.0.1
---
排错命令（出问题时）

# 客户端：确认 SSH 落到哪台机器
ssh -p 34134 -i /data2/liujingzhi/id_ed25519_5090 \
  root@116.63.180.90 "hostname"
# 客户端：从 SSH 入口机访问 0005 的服务
ssh -p 34134 -i /data2/liujingzhi/id_ed25519_5090 \
  root@116.63.180.90 "curl -s http://192.168.0.138:8000/he
