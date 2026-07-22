# Dobot VLA Server–Client

用于在 GPU 服务器运行 Base PI-0.5 Dobot VLA，并通过 SSH 隧道连接 Dobot 真机客户端。

## 1. 登录 GPU 服务器

```bash
ssh -J e85f315bde7748bab59f40bcf30642f5@proxy.nscc-gz.cn:8022 \
  sysu_xdliang_2@pytorch-ng-1984438-lfwj
```

## 2. 启动服务

### Base VLA

```bash
cd /HOME/sysu_xdliang/sysu_xdliang_2/HDD_POOL/amsong/rlt-openpi
./server/serve_dobot_vla.sh "" 8000
```

## 3. 建立 SSH 隧道

在 Dobot 客户端执行并保持终端运行：

```bash
ssh -J e85f315bde7748bab59f40bcf30642f5@proxy.nscc-gz.cn:8022 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -N \
  -L 127.0.0.1:18000:127.0.0.1:8000 \
  sysu_xdliang_2@pytorch-ng-1984438-lfwj
```

## 4. 检查连接

```bash
cd /home/sbc/dobot_xtrainer
python experiments/run_ws_inference.py \
  --ws-host 127.0.0.1 \
  --ws-port 18000 \
  --check-only
```

## 5. Dry run

```bash
cd /home/sbc/dobot_xtrainer
python experiments/run_ws_inference.py \
  --ws-host 127.0.0.1 \
  --ws-port 18000 \
  --robot-port 6001 \
  --hostname 127.0.0.1 \
  --instruction "pour water" \
  --action-chunk-len 1 \
  --dry-run
```

## 6. 真机运行

```bash
cd /home/sbc/dobot_xtrainer
python experiments/run_inference_with_intervention.py \
  --ws-host 127.0.0.1 \
  --ws-port 18000 \
  --robot-port 6001 \
  --hostname 127.0.0.1 \
  --instruction "pour water"
```
