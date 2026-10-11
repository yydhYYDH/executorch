# 最小整数 HMX 矩阵乘与 FP32 输出

2026-10-11，**device tier**：在 PJZ110 / SM8750 / Hexagon v79 手机上验证。
独立 DSP `.so`，不依赖 QNN、ExecuTorch 模型或 HexKL。
生产后端没有改动，没有运行模拟器。

固定形状 `A[64,32] @ B[32,64]`，输入和输出接口都是 row-major：

| 张量 | 类型 | scale | zero point |
|---|---|---:|---:|
| A | uint8 | 1/8 | 128 |
| B | int8 | 1/4 | 0 |
| Q | uint8 | 1/32 | 128 |
| Y | FP32 | — | — |

```text
acc[r,n] = sum_k((A[r,k]-128) * B[k,n])
Q[r,n]   = clamp(acc[r,n]+128, 0, 255)
Y[r,n]   = (Q[r,n]-128) / 32
```

Y 是反量化后的量化矩阵乘结果；超过 Q 的范围会饱和，所以不能将 Y 当作
始终等于无饱和 FP32 矩阵乘。例子直接接收已量化的 A/B，没有 FP32 输入量化步骤。
选取 `scale_A * scale_B = scale_Q`，使 requantization 倍率为 1，省去一般倍率
和舍入规则，便于观察整数计算与输出。

## 实现

- `matmul.c`：标量 pack → HMX 内联汇编 → 标量 unpack / FP32 反量化。
- `main.c`：HMX 上电、申请 16 KiB VTCM、加锁、三组完整数值检查、输出文件、释放资源。
- `build.sh`：只编译这两个 C 文件并导出反汇编。
- `run_device.py`：等待 llama 工作负载、SSH 部署与 hash 检查、执行、取回完整输出，
  用主机 NumPy 独立检查全部元素。

核心指令是实际整数 HMX，不是用 FP16 模拟整数乘法：

```asm
bias = mxmem2(...)
bias = mxmem2(...)
mxclracc
{ activation.ub = mxmem(...,2047):deep
  weight.b = mxmem(...,2047):deep }
cvt.ub = acc(0)
mxmem(...) = cvt
cvt.ub = acc(0x1001)
mxmem(...) = cvt
```

第二次转换读取另外 32 列并选择第二套转换参数。这里列出的是该固定例子验证过的
配置，不是所有位域的完整定义。`0x40006000` 配合第二个 128-byte 向量中的
`-128 * sum_k(B[k,n])`，在此例中实现单位倍率、输出 zero point 128、输入零点修正。
整数转换表不同于 FP16 scale/bias 表，不能把 `0x6000` 当作 FP16 scale。

### FP32 反量化的实际指令

`matmul.c` 的 `y = ((int)value - 128) * (1.0f / 32.0f)` 编译为以下标量
Hexagon 指令。它们来自当前设备验证过的 `.so` 的反汇编，按数据依赖顺序摘录；
省略了地址计算、循环以及同 packet 的其他指令，不能把这段直接当作完整 packet 调度。

```asm
// r4 = 0x3d000000，FP32 的 1/32；在循环外设置。
// 库内偏移     指令
// 0xadc
r13 = memub(r5+r14<<#0x0)
// 0xae0
r28 = add(r13,#-0x80)
// 0xaec
r28 = convert_w2sf(r28)
// 0xb00
r13 = sfmpy(r28,r4)
// 0xb04，与 sfmpy 同 packet，消费 new value。
memw(r6++#0x4) = r13.new
```

| 指令 | 在本例中的作用 |
|---|---|
| `memub` | 从 native 输出读取 uint8，并零扩展为寄存器整数 |
| `add(...,#-0x80)` | 减去输出 zero point 128 |
| `convert_w2sf` | signed int32 转成 FP32 |
| `sfmpy` | FP32 乘输出 scale 1/32 |
| `memw(...)=r13.new` | 保存 FP32 位模式到 row-major 输出 |

即 `Y = (Q - zero_point) * scale`。本例反量化不是一条专门的
“dequantize”指令，而是整数转浮点与浮点乘法的组合，全部在 DSP 上执行。
HMX `cvt.ub = acc(...)` 是此前的**重定量化**步骤，输出仍为 uint8，
不能将它称作 uint8 到 FP32 的反量化。两者之间还有 native 输出布局的读取和转换。

反汇编证据：`.tmp/scratch/mnv2/hmx_int8_minimal/example/example.asm`，
设备验证的 `.so` SHA256：
`9b6161d03d8972fdb92cbec5d48fe54406ff2ed687ff3730175681d4a95107dc`。
偏移仅针对这份二进制。SDK 定义可查 tools 19.0.04 的
`target/hexagon/include/hexagon_protos.h` 中 `convert_w2sf`。

后续可用 HVX 的 uint8 widening/unpack、整数减法、整数到浮点转换和浮点乘法
批量实现，并与布局转换合并；当前例子没有实现或验证该向量版本。
这些标量指令是本例的证据，不代表已经捕获 QNN 自身的反量化函数。

VTCM workspace 至少 12,288 字节，基址 2048-byte 对齐：

| 偏移 | 内容 | 大小 |
|---:|---|---:|
| 0 | packed A | 2048 B |
| 2048 | packed B | 2048 B |
| 4096 | 两组 native uint8 输出 | 4096 B |
| 8192 | 两套转换表 | 512 B |

本例普通 `:deep` 变体的 byte index：

```text
A(r,k) = (r/4)*128 + k*4 + r%4
B(k,n) = (n/32)*1024 + (k/4)*128 + (n%32)*4 + k%4
Q(r,n) = (n/32)*2048 + (r/4)*128 + (n%32)*4 + r%4
```

尤其是 B 的两个 32 列组各自连续，不能按 K 的四行组交替存储它们。
全 1 输入会掩盖这种布局错误。本次用独立 `.so` 的单位置激活/权重探针和完整
变化值矩阵验证了布局。QNN 实际命中的 `:deep:cm` 路径仍是另一变体；
本例不宣称复制了 QNN 全部布局或调度。

## 复现

本工作区执行，Python 需安装 NumPy，SSH `oneplus13-reverse` 可以连接手机。
日志捕获用已授权的 `su -c logcat`；不清空系统日志、不终止用户进程。

```bash
cd /home/yydh/executorch
.tmp/locks/lockrun heavy -- bash backends/hexagon/examples/hmx_int8_minimal/build.sh
.tmp/locks/lockrun device -- /home/yydh/miniconda3/envs/et/bin/python backends/hexagon/examples/hmx_int8_minimal/run_device.py
```

默认 SDK `/home/yydh/hexagon-tutorial/tools/hexagon-sdk`，tools 19.0.04。
构建路径可通过 `HEXAGON_SDK_ROOT`、`HEXAGON_TOOLS_ROOT`、
`HMX_EXAMPLE_BUILD_DIR` 修改；执行脚本对应提供 `--sdk`、`--build-dir`、
`--host` 和 `--remote-dir`。端到端换 SDK 后需重新验证。

产物默认在 `.tmp/scratch/mnv2/hmx_int8_minimal/example/`：

- `output.u8`：三组 `[64,64]` uint8，按 case 顺序连续存储。
- `output.f32`：同顺序的 little-endian FP32。
- `checks.json`：独立主机验收结果、完整输出 hash、实际 `.so` hash。
- `device.log`、`example.farf.log`、`device_hashes.txt`、`example.asm`：原始证据。

执行会覆盖该私有示例目录中的同名输出。SDK runner 的进程退出码只表示 RPC
状态，脚本另检查 `main() ... return value of 0`，并核对完整输出；不能只看 exit 0。

## 设备结果

| case | 输入特点 | Q 错误元素 | FP32 错误元素 | 饱和元素 |
|---|---|---:|---:|---:|
| 0 | A 围绕 zero point 变化，B 正负值 | 0/4096 | 0/4096 | 169 |
| 1 | A 覆盖 uint8，B 正负值，更大的累加结果 | 0/4096 | 0/4096 | 3852 |
| 2 | A 全是 zero point，B 非零 | 0/4096 | 0/4096 | 0 |

case 0 第一行前八个输出：

```text
Q = [132, 98, 124, 130, 96, 132, 98, 124]
Y = [0.125, -0.9375, -0.125, 0.0625, -1.0, 0.125, -0.9375, -0.125]
```

完整 Q SHA256：`4e28c3703a7a86dd433f596c7fcab8b5c5dd094d1bcb6a4d45af782e27ce75f9`。
完整 Y SHA256：`c63f32fb971450af9aac2429113c0e839c0989ebce4c9309c809d72a60cc07ee`。
数据布局和整数转换只覆盖这个固定 tile 与固定量化参数。该实现用于演示，
没有 HVX pack、DMA 流水、多 tile K 累加或性能调优，没有与 QNN 做速度比较。

QNN 自身的整数入口证据另见 [QNN_INT_MATMUL_DISPATCH.md](../../QNN_INT_MATMUL_DISPATCH.md)。
