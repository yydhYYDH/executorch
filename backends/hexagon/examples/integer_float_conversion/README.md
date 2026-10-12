# INT4 / INT8 / FP16 / FP32 独立转换示例

不依赖 QNN 或 HMX。相同 C 源码可以在主机或 Hexagon DSP 上调用。
[conversion.h](conversion.h) 定义 API；[conversion.c](conversion.c) 实现转换；
[main.c](main.c) 展示每个方向的调用和二进制输出。

## 接口与语义

`quantize_sint(x,count,bits,scale,zp,q)` 支持 bits=4 或 8。
使用 FP32 除法和加法，`Q=clamp(round_nearest_even(X/scale+zp))`。
INT4 范围 -8..7，INT8 范围 -128..127，结果均先存在 `int8_t` 数组中。
`dequantize_sint` 计算 FP32 `X=(Q-zp)*scale`。

`requantize_sint` 接受两侧独立的位宽、scale 和 zero point，先按源参数
反量化到 FP32，再按目标参数量化。INT8→INT4 不等于取低四位，尤其在两侧
量化参数不同时。这里是清楚展示语义的标量参考实现，不是性能调优结果。

`pack_int4` 使用两个有符号二补码 nibble 一个字节：偶数序号占低四位，
奇数占高四位。奇数长度的末字节高四位补零。比如 `[-8,-1,0,7,1]` 打包为
`[0xf8,0x70,0x01]`；解包恢复原来的有符号数。
这个线性布局不等于 HMX 权重 tile 布局。

`fp32_to_fp16_bits` 返回 IEEE binary16 原始位模式，以 `uint16_t` 存储；
`fp16_bits_to_fp32` 扩展为 FP32。FP16 本身没有 scale/zero point。
FP32→FP16 会发生浮点舍入，超范围可以得到 infinity；FP16→FP32 可以精确
表示有限 binary16 值。NaN 保证仍是 NaN，不保证 payload 不变。

所有 affine API 要求正有限 scale 和位于目标整数范围内的 zero point。
无效参数、NaN 量化输入、超范围的逻辑整数返回 1，成功返回 0。
错误返回前不写目标。无穷量化输入饱和到上下端点。数组应提供足够空间；
除 `requantize_sint` 可使用相同 src/dst 数组外，输入与输出不能重叠。
`pack_int4` 的目标容量为 `(count+1)/2` 字节。

## 组合转换

INT8→FP16；INT4 也一样，改 bits=4：

```c
float temporary[32];
uint16_t half_bits[32];
int status = dequantize_sint(q, 32, 8, scale, zp, temporary);
if (status == 0) fp32_to_fp16_bits(temporary, 32, half_bits);
```

FP16→INT4；INT8 改 bits=8：

```c
float temporary[32];
int8_t logical_int4[32];
uint8_t packed_int4[16];
fp16_bits_to_fp32(half_bits, 32, temporary);
int status = quantize_sint(temporary, 32, 4, scale, zp, logical_int4);
if (status == 0) status = pack_int4(logical_int4, 32, packed_int4);
```

如果希望复刻 `use_fp16a8w` 的权重准备，还需先把 scale 舍入为 FP16，再乘
INT8 并舍入为 FP16；参见 [FP16A8W 矩阵乘](../hmx_fp16a8w_minimal/README.md)。
本示例默认 affine 反量化使用原 FP32 scale，两者的运算顺序不同。

## 构建和验证

从仓库根目录执行：

```bash
CONVERT_PY=/home/yydh/miniconda3/envs/et/bin/python
.tmp/locks/lockrun heavy -- bash backends/hexagon/examples/integer_float_conversion/build.sh
"$CONVERT_PY" backends/hexagon/examples/integer_float_conversion/verify.py
.tmp/locks/lockrun device -- "$CONVERT_PY" backends/hexagon/examples/integer_float_conversion/run_device.py
```

`CONVERSION_HOST_CC` 可指定支持 `_Float16` 的主机 Clang；默认用本机 NDK
Clang。DSP 编译器为 Hexagon tools 19.0.04，目标 v79。
可用 `HEXAGON_SDK_ROOT`、`HEXAGON_TOOLS_ROOT`、`CONVERSION_BUILD_DIR`
覆盖构建环境；运行脚本支持 `--host/--sdk/--build-dir/--remote-dir`。
SDK launcher 启动 DSP 标量程序，不需要 HMX lease 或 VTCM。

Scratch：`.tmp/scratch/mnv2/integer_float_conversion/`。
主机已通过 20 组测试、8 组非法输入拒绝测试及全部 65,536 种 FP16 位模式。
测试覆盖零长度、奇数长度、负数、舍入中点、饱和、zero point 边界、FP16
subnormal、signed zero、infinity 和 NaN。浮点结果按位比较；NaN 仅比较类别。
手机 PJZ110 / SM8750 / HTP v79 上同样通过 20 组、全部 65,536 个 FP16
位模式；见 `device_checks.json`、`device.log`、`device_hashes.txt` 和
`device_output.bin`。主机非法输入检查单独记录在 `host_checks.json`。
设备 SO SHA256：`b18fde75e9949d2e11d30c43d89aa88e47ac4a522dc9f5e095d264e31a2cb0e5`。
设备输出 SHA256：`84bb0b82ee7d7b8ac5c63782b6a2f15ff61a353c41b6a02dafab576143cd33a0`。

SDK 解码的 `conversion.asm` 可查看整数/浮点转换。
当前构建使用 `convert_w2sf`、`convert_sf2w:chop` 等标量整数指令，FP16
转换由编译器调用 `__truncsfhf2` / `__extendhfsf2` 实现。
这里没有使用 HMX `cvt.uh/cvt.ub/cvt.hf`：那些指令转换的是矩阵累加器，
不能直接当作任意内存张量的通用类型转换。
