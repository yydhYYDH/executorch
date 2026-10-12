# 整数矩阵乘与类型转换示例索引

这些是可以独立调用的固定 tile 示例，已提交在 `da48dc65`。每个 README
提供构建、DSP 运行、布局、参数范围和数值证据。不是通用任意形状算子。

## 配置与示例对应

| QNN 配置 | 激活 | 存储/逻辑权重 | HMX 转换后输出 | 内核族 | M/K/N | 独立示例 / 调用入口 |
|---|---|---|---|---|---|---|
| `use_16a16w` | UINT16 | INT16 | UINT16 | `hhh` | 32/32/64 | [hmx_16a16w_minimal](hmx_16a16w_minimal/README.md)，`hmx_matmul_a16w16_tile_quantized` |
| `use_16a8w` | UINT16 | INT8，已中心化 | UINT16 | `hbh` | 32/32/64 | [hmx_16a8w_minimal](hmx_16a8w_minimal/README.md)，`hmx_matmul_a16w8_tile_quantized` |
| `use_16a4w` | UINT16 | INT4 | UINT16 | `hnh` | 32/32/64 | [hmx_int4_minimal](hmx_int4_minimal/README.md)，`hmx_matmul_u16s4_tile_quantized` |
| `use_8a8w` | UINT8 | INT8 | UINT8 | `bbb` | 64/32/64 | [hmx_int8_minimal](hmx_int8_minimal/README.md)，`hmx_matmul_u8s8_tile_quantized` |
| `use_8a4w` | UINT8 | INT4 | UINT8 | `bnb` | 64/32/128 | [hmx_int4_minimal](hmx_int4_minimal/README.md)，`hmx_matmul_u8s4_tile_quantized` |
| `use_fp16a8w` | FP16 | INT8，先反量化成 FP16 | FP16，示例再扩展到 FP32 | `f16` | 32/32/64 | [hmx_fp16a8w_minimal](hmx_fp16a8w_minimal/README.md)，`hmx_matmul_fp16a8w_tile` |
| `use_16a2w` | UINT16 | INT2 | UINT16 | `hch` | 32/32/64 | [hmx_16a2w_minimal](hmx_16a2w_minimal/README.md)，`hmx_matmul_u16s2_tile` |
| `use_16a4w_block` | UINT16 | INT4 × block multiplier，准备成 INT8 | UINT16 | `hbh`，LPBQ | 32/128/64 | [hmx_lpbq_minimal](hmx_lpbq_minimal/README.md)，`hmx_matmul_u16s4_lpbq_tile` |

A8W4 与 A16W4 共用一个目录，但各自有独立、可直接调用的函数。
整数例子的量化输出可按 `Y=(Q-zq)*sq` 反量化为 FP32；A16W8/A16W16 API
直接返回 UINT16 Q，调用者完成这一步，其余例子也提供 Y 输出。
共享系数生成及可变参数测试见 [hmx_quant_params](hmx_quant_params/README.md)。

## 内核名字与真实指令

`hhh/hbh/hnh/bbb/bnb` 描述 QNN 内核族的激活、权重和转换后输出元素
类型：`h` 为 16-bit 整数 halfword，`b` 为 8-bit byte，`n` 为 4-bit nibble；
INT2 路径的 `c` 对应 2-bit 权重。名字本身不编码 signed/unsigned，也不是
逐字对应 HMX 助记符的寄存器声明。

例如，A16 路径的汇编也可能写 `activation.ub`，UINT16 激活的实际布局和
加载宽度由 selector 决定；INT16 权重示例用拆分字节与 `weight.b:dilate`
加载。要核对类型，应看实际输入布局、selector、指令 modifier 和设备数值，
不能只看 `ub/b` 的拼写。这里的整数 `h` 不等于浮点 `hf`。

`use_fp16a8w` 的已验证路径使用 `activation.hf / weight.hf / cvt.hf`。
权重先按 `FP16(INT8 * FP16(scale))` 准备。它没有整数输出 zero point；
最后的 FP32 是 FP16 输出的扩展，不是一次 FP32 精度矩阵乘。

## INT4 / INT8 / FP16 / FP32 转换

新增独立目录 [integer_float_conversion](integer_float_conversion/README.md)：

| 转换 | 调用方式 |
|---|---|
| FP32 → INT4 / INT8 | `quantize_sint`，指定 bits=4/8、scale、zero point |
| INT4 / INT8 → FP32 | `dequantize_sint` |
| INT8 ↔ INT4 | `requantize_sint`，允许两侧使用不同 scale / zero point |
| INT4 逻辑值 ↔ 两值一字节 | `pack_int4` / `unpack_int4` |
| FP32 ↔ FP16 | `fp32_to_fp16_bits` / `fp16_bits_to_fp32` |
| INT4 / INT8 ↔ FP16 | 组合上面的反量化/量化与 FP16 转换；运行示例包含两方向校验 |

这里的 INT4 打包是普通线性 nibble 容器，偶数元素放低半字节，奇数元素放高
半字节；**不是 HMX 矩阵 tile 布局**。矩阵乘示例仍由自己的 packer 生成对应
HMX 布局。普通 affine 量化采用 nearest-even 舍入和饱和，不能替代 HMX 的
`acc → cvt` 重定量化系数/截断规则。
