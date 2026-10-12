# Paraformer 会议转写复现指南（VAD + Paraformer + CAM++）

本目录是给**另一个 agent / 另一位开发者**的完整复现包：从零下载依赖和模型，
导出全部 `.pte`，构建 runner，部署到手机（OnePlus 13 / Hexagon），
最后跑**离线**（整段录音）和**流式**（TCP + 浏览器）两条链路。

整个系统的三个模块：| 模块 | 跑在哪 | 产物 |
|---|---|---|
| **FSMN-VAD**（端点检测） | Hexagon DSP | `fsmn_vad_enc_T20.pte` |
| **Paraformer**（流式 ASR） | Hexagon DSP | `encp_one.pte` + `pred15.pte` + `decoder16_first.pte` + `decoder16_next.pte` |
| **CAM++**（说话人 / diarization） | 主机 CPU（XNNPACK） | `campplus_cpu.pte` |

> **务必先读 `../../examples/paraformer/README.md` 和 `DEPLOY.md`**：那两份是
> runner 的协议与部署说明，本文件只负责「怎么从零把东西造出来」。

---

> **路径约定**：`export/ calib/ deploy/ run/ config*.sh reproduce.sh` 相对本目录
> `examples/paraformer/repro/`；`examples/...` 相对仓库根。

## 0. 目录

```
examples/paraformer/repro/
  README.md                  ← 本文件
  config.example.sh          ← 复制成 config.sh，填路径
  reproduce.sh               ← 一键：下载模型 + 标定 + 导出 PTE
  download_models.py         ← 从 ModelScope 拉三个模型
  export/
    hexagon_core.py          ← encoder 的 Hexagon 包装（缩放链核心）
    calibrate.py             ← 生成 alpha_fc / fc_inmag / ffn_mag_fc.json
    make_example_inputs.py   ← 生成导出用的 trace 输入
    export_encoder.py        ← 导出 encp_one.pte（含 fp16 clamp）
    export_predictor.py      ← 导出 pred15.pte
    export_decoder.py        ← 导出 decoder16_{first,next}.pte
    export_campplus.py       ← 导出 campplus_cpu.pte
    campplus_model.py        ← CAM++ 网络定义（随仓库分发）
  calib/                     ← 已标定好的常量（也可用 calibrate.py 重算）
    alpha_fc.json  fc_inmag.json  ffn_mag_fc.json
  deploy/
    push_assets.sh           ← 把 runner + 全部资产推到手机
  run/
    offline.sh               ← 整段录音，设备上一把过
    stream.sh                ← 起 TCP 服务 + 反向隧道 + 桥
    frontend.sh              ← 起浏览器前端
```

---

## 1. 环境准备

### 1.1 主机（导出 + 桥接）

- Linux x86_64，Python 3.10+。
- 克隆 ExecuTorch 并装好 Python 包（见仓库根 `AGENTS.md` / `docs/`）：
  ```bash
  git clone <this-repo> executorch && cd executorch
  ./install_executorch.sh          # 或 pip install -e . --no-build-isolation
  ```
- 装 FunASR（用于加载模型、导出与对拍）：
  ```bash
  pip install funasr modelscope torch torchaudio soundfile
  # 注意：np 版本要与 executorch 一致；onnx 相关可不装
  ```
- 本仓库当前分支为 Hexagon 后端分支：`backend-hexagon-minimal`。
  Hexagon 的 Python 导出接口（`executorch.backends.hexagon.*`）必须在
  这个分支上，`main` 上没有。

### 1.2 手机（运行）

- OnePlus 13（骁龙 8 Elite，Hexagon v79），已 root，Termux 环境。
- 手机目录约定（下同）：`/data/data/com.termux/files/home/csm/execuTorch-ds`。
- ssh 别名建议写成 `oneplus13-reverse`（见 `~/.ssh/config`）：
  ```
  Host oneplus13-reverse
      HostName <phone-ip>
      User <user>
      Port <port>
  ```
  > 该机 ssh 失败率偏高，脚本里都带了重试；这是已知现象，不是配置错。

### 1.3 服务器（可选，公网前端）

- 一台能跑 Python 的服务器（本项目用 `chengdu`），systemd + nginx。
- 详见 `DEPLOY.md`。

---

## 2. 下载模型

```bash
python download_models.py --into ~/models
```

拉三个 ModelScope 仓库：

| 用途 | 仓库 |
|---|---|
| Paraformer 流式 | `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online` |
| FSMN-VAD | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` |
| CAM++ | `iic/speech_campplus_sv_zh-cn_16k-common` |

（`download_models.py` 用 `modelscope.snapshot_download`，首次会缓存到 `~/.cache/modelscope`。）

---

## 3. 标定（得到 encoder 的逐层缩放常量）

encoder 的 fp16 缩放链依赖三张**逐层幅度表**：块输入幅度 `M`、块输出幅度
`alpha`、FFN 输出幅度 `betas`。它们由模型自己跑一遍 example wav 量出来：

```bash
python export/calibrate.py \
  --model-dir "$PARAFORMER_DIR" \
  --calib-dir calib
```

`repro/calib/` 里已附带**同一 checkpoint** 标定出的三张表，所以正常情况
这一步可以跳过（导出结果应与 `encp_one_min.pte` **逐字节相同**）。
换模型 / 换权重时才需要重算。

> **为什么需要标定**：Paraformer encoder 是 50 层、残差幅度逐渐放大，
> 直接塞进 fp16 DSP arena 会在深层的 q·kᵀ 或 FFN 溢出到 `inf`。
> `hexagon_core.py` 按逐层幅度把每层输入 `/M[j]`、输出 `*alpha`，把整条链
> 压在 fp16 范围内；`torch.minimum(scores, 65504)` 是最后一道保险
> （见 §5）。

---

## 4. 导出全部 PTE

### 4.1 VAD（Hexagon DSP）

```bash
python examples/fsmn_vad/export_fsmn_vad.py "$FSMN_VAD_DIR" \
  --chunk 20 --out out/fsmn_vad_enc_T20.pte
python examples/fsmn_vad/prepare_assets.py "$FSMN_VAD_DIR" out/fsmn
# out/fsmn/cmvn.bin + tokens.json
```

### 4.2 Paraformer encoder（Hexagon DSP，含 fp16 clamp）

```bash
python export/export_encoder.py \
  --model-dir "$PARAFORMER_DIR" \
  --calib-dir calib \
  --out out/encp_one.pte \
  --diag-dir out/diag
```

脚本会：
1. 用 `make_example_inputs.py` 生成 4 个真实 chunk 输入；
2. exec `hexagon_core.py` 包装 encoder；
3. 验证 **mask 改写后 host 输出逐位不变**、**export 前后逐位不变**；
4. lowering 成单 delegate 的 `.pte`。

产物即固定版 encoder（对应的 `encp_one_min.pte`）。
> 旧的、未 clamp 的 `encp_one.pte` 在响亮的远场会议里会在 score matmul 溢出
> fp16，污染 streaming cache。**上机必须用本脚本产出的这个**。

### 4.3 Paraformer predictor / decoder

```bash
python export/make_example_inputs.py --model-dir "$PARAFORMER_DIR" --out-dir out/diag   # 若 4.2 已跑过可跳过
python export/export_predictor.py --model-dir "$PARAFORMER_DIR" --out out/pred15.pte
python export/export_decoder.py   --model-dir "$PARAFORMER_DIR" \
  --input out/diag/decoder_input.bin --first --out out/decoder16_first.pte
python export/export_decoder.py   --model-dir "$PARAFORMER_DIR" \
  --input out/diag/decoder_input.bin --out out/decoder16_next.pte
```

> **注意**：`--first` 与不带 `--first` 是**两个不同的图**（前者建 cache，
> 后者读 cache），两个都要。二者输入 buffer 形状相同。

### 4.4 前端资源

```bash
python examples/paraformer/prepare_assets.py "$PARAFORMER_DIR" out
# out/cmvn.bin + out/tokens.json
```

### 4.5 CAM++（CPU / XNNPACK）

```bash
python export/export_campplus.py \
  --checkpoint "$CAMPP_PLUS_DIR/campplus_cn_common.bin" \
  --out out/campplus_cpu.pte
```

### 4.6 一键

```bash
cp config.example.sh config.sh   # 填三个模型目录
bash reproduce.sh
```

---

## 5. 关键正确性约束（踩过的坑）

1. **encoder 必须用 clamp 版**。score matmul 是唯一溢出点，`torch.minimum
   (s, 65504)` 是 `a<b?a:b`；**不能用 DSP 的 unary CLAMP**——它按设计让
   `inf` 通过（`unary_ops.cc:534`）。
2. **VAD 的 250 宽线性层要 pad 到 256**（`export_fsmn_vad.py` 默认做），
   否则 DSP HMX matmul 报 `0x8000040d`。
3. **decoder 的 16 个 token 槽**是上限；predictor 一 chunk 超 16 会报错。
   chunk = 600 ms。
4. **mask 改写**：attention 的 `eq -> masked_fill` 链在 Hexagon 上没有 `eq`
   kernel，`hexagon_core.py` 把它改写成 `where`（`FUSE_MASK`），并把位置编码、
   valid mask 预算到 host 作为图输入（`PREPARED_ENCODER`）。改完必须
   逐位自检（脚本已内置 assert）。
5. **证据纪律**：`rc=0` 不代表数值对；`scp/rsync` 会静默失败，推完核对
   `md5sum`。

---

## 6. 构建 runner

runner 是单个 C++ target：`paraformer_runner`（含 VAD + speaker）。

```bash
# 主机（x86，跑不了 Hexagon，用来验证逻辑 / 前端联调）
cmake -B cmake-out -DEXECUTORCH_BUILD_PARAFORMER_RUNNER=ON
cmake --build cmake-out --target paraformer_runner -j

# 交叉编译到 Android（Hexagon 后端）见仓库 root CMakePresets / docs：
#   docs/source/using-executorch-building-from-source.md
#   以及 backends/hexagon 的 Android 构建说明
```

VAD-only 的 `fsmn_vad_runner` 用 `-DEXECUTORCH_BUILD_FSMN_VAD_RUNNER=ON`。

设备上约定文件名：把 `paraformer_runner` 拷成 `paraformer_runner_spk`
（`start_tcp_service.sh` 优先用它），并准备 `skel/`（与 `.pte` 同版本编出的
Hexagon skel）。

---

## 7. 部署到手机

```bash
cp config.example.sh config.sh   # 填 PHONE / PHONE_DIR
bash deploy/push_assets.sh
```

或手动，设备目录里要有：

| 资产 | 说明 |
|---|---|
| `paraformer_runner_spk` | VAD + speaker runner |
| `encp_one.pte` | clamp 版 encoder |
| `pred15.pte` `decoder16_first.pte` `decoder16_next.pte` | predictor/decoder |
| `cmvn.bin` `tokens.json` | Paraformer 前端 / 词表 |
| `fsmn/fsmn_vad_enc_T20.pte` `fsmn/cmvn.bin` | VAD |
| `camppluscpu/campplus_cpu.pte` | CAM++ speaker |
| `skel/` | Hexagon skel |

---

## 8. 离线：整段会议录音

设备上直接跑（runner 在设备算 fbank/LFR/CMVN，逐 chunk 走三块 PTE）：

```bash
ssh "$PHONE" "cd '$PHONE_DIR' && \
  LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 ADSP_LIBRARY_PATH=\$PWD/skel \
  ./paraformer_runner_spk --audio meeting.wav \
    --cmvn cmvn.bin --tokens tokens.json \
    --model_path encp_one.pte --predictor pred15.pte \
    --decoder_first decoder16_first.pte --decoder_next decoder16_next.pte \
    --vad_model fsmn/fsmn_vad_enc_T20.pte --cmvn_vad fsmn/cmvn.bin \
    --spk_model camppluscpu/campplus_cpu.pte"
```

输出：`TEXT ...` 整段文本，以及 `SEG start end SPK<n> text` 逐段带说话人。

也可用主机 Python 驱动三块 PTE（离线和设备跑同一条数值链，便于逐步对拍）：

```bash
python examples/paraformer/paraformer_streaming.py meeting.wav \
  --model-dir "$PARAFORMER_DIR" \
  --encoder out/encp_one.pte --predictor out/pred15.pte \
  --decoder-first out/decoder16_first.pte --decoder-next out/decoder16_next.pte \
  --remote "$PHONE" --device-dir "$PHONE_DIR" \
  --checkpoint meeting.npz
```

或用 `run/offline.sh`。

---

## 9. 流式：TCP + 实时

设备端起 TCP 服务（只绑 `127.0.0.1`）：

```bash
ssh "$PHONE" "cd '$PHONE_DIR' && PORT=9000 ./start_tcp_service.sh"
```

反向隧道（手机 → 服务器 loopback；若只在本地玩，隧道可省，直连手机端口需
把 `server.py` 指向手机）：

```bash
autossh -M 0 -NT -R 127.0.0.1:9000:127.0.0.1:9000 "$SERVER"
```

服务器上加桥：

```bash
python3 examples/paraformer/web/server.py --runner-port 9000 --listen-port 8020
```

TCP 协议（见 README）：`0x81` 部分文本、`0x83` 分段+说话人、
`0x84` 最终 diarization、`0x82` 最终结果。见 `run/stream.sh`。

---

## 10. 浏览器前端

```bash
python3 examples/paraformer/web/server.py --runner-port 9000 --listen-port 8020
# 打开 http://127.0.0.1:8020/
```

- 麦克风实时录音 → 实时部分文本 + 每句一行、按说话人配色；
- `diarization` 事件到达后用离线聚类标签重绘；
- 文件上传会解码成 16 kHz 单声道走同一路径。

公网（nginx + systemd + HTTPS）见 `DEPLOY.md`；关键点 **`proxy_buffering off`**，
否则 SSE 的 partial/segment 会被 nginx 缓冲住。

或用 `run/frontend.sh`。

---

## 11. 验收基线（同一测试音频）

在 OnePlus 13 上实测（供复现时判断是否「跑对」）：

| 指标 | 值 |
|---|---|
| 纯 ASR RTF | ~0.204 |
| VAD + ASR RTF（120 s / 39 min） | ~0.211 / ~0.221 |
| + speaker RTF | ~0.222 |
| 流式 TCP RTF | ~0.272–0.281 |
| VAD RTF | ~0.009 |
| CAM++ 单段耗时 | ~30 ms / 3 s |
| diarization AHC（6 min / 39 min） | ~0.920 / ~0.880 |

数值正确性自检（导出脚本已内置）：
- encoder：mask 改写前后逐位相等；export 前后逐位相等；
- predictor / decoder：host eager 与 exported 逐位相等。

---

## 12. 与真实会议音频对齐（可选）

收集 `TextGrid` / `RTTM` 参考，和 runner 的 `SEG` 行对拍：

```
examples/paraformer/repro/run/offline.sh meeting.wav     # 产出 transcript + SEG
```

参考：39 min 会议（AISHELL-4，单通道 far-field）整段 CER ~41.5%，
非重叠段 CER ~36.7%；远场、重叠语音、无语言模型 LM 是主要误差源。
这些数字是这套**无 LM 流式系统**的正常水平，不是 bug。
