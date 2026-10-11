# QNN 整数矩阵乘：设备确认入口与分支

2026-10-11，device evidence。PJZ110 / SM8750 / HTP v79，QNN
2.41.0.251128；分支 `backend-hexagon-minimal`。没有修改生产算子或公共QNN库，
没有运行模拟器。本记录与此前FP16的57形状速度表独立。

## 已确认的结果

测试两种图：静态权重 `Linear(bias=False)` 经QNN FullyConnected builder，
以及两个运行时输入的 `A @ B` 经QNN MatMul builder。两者数学上都是矩阵乘，
但不能将它们的内部调度视为相同。

Linear使用PTQ A8W8：uint8激活、int8静态权重，per-tensor量化。
MatMul使用默认8位配置：两个输入的QDQ dtype都是uint8。
模型接口均为FP32，输入量化、整数计算、输出反量化全部处于同一个QNN delegate；
边界图没有外部CPU计算节点。`export_matmul.log`确认实际访问
`aten.matmul.default`，不是把双输入MatMul写成静态Linear。

五模型均捕获到：

```text
已归一化的调用packet 0x3a85c8
  call hmx_v73_convbbb1x1_stride1              0x2a7d80
    params[12] bit5 = 1
    jump hmx_v73_convbbb1x1deep_stride1       0x2a8e00
```

| 图 | M×K×N | 每次Graph执行的入口调用数 | deep分支调用数 | 四次的deep累积计数 |
|---|---|---:|---:|---|
| 静态Linear，A8W8 | 128×128×128 | 1 | 1 | 1 / 2 / 3 / 4 |
| 静态Linear，A8W8 | 1024×1024×1024 | 80 | 80 | 80 / 160 / 240 / 320 |
| 静态Linear，A8W8 | 1×1024×1024 | 4 | 4 | 4 / 8 / 12 / 16 |
| 双输入MatMul，uint8/uint8 QDQ | 128×128×128 | 1 | 1 | 1 / 2 / 3 / 4 |
| 双输入MatMul，uint8/uint8 QDQ | 1024×1024×1024 | 16 | 16 | 16 / 32 / 48 / 64 |

这里是**函数调用次数**，不是HMX指令条数、DMA并发数或线程数。
每个模型均有原版前、计数探针、恢复原版后三个新runner进程，每进程4次执行。
该最终系列共15个成功进程、60次Graph执行；每模型三个完整最终输出逐位一致。
五份记录见 `.tmp/scratch/mnv2/qnn_int_matmul/dispatch_checks.json`。

1024³的两种图入口相同、调用次数不同，首个输出描述符也不同。这可以确认外围
分块有区别，不能仅据80与16推出速度比或判断哪种调度更好。

## 解码出的整数指令与实参

SDK objdump使用 `--mcpu=hexagonv79 --mattr=+hmx`。
已命中deep函数包含：

```asm
activation.ub = mxmem(r6,r7):deep:cm
weight.b = mxmem(r8,r9):deep
bias = mxmem2(r3)
cvt.ub = acc(r25)
mxmem(r10,r11):cm = cvt
```

它与FP16的 `activation.hf / weight.hf / cvt.hf` 路径不同。
双输入MatMul的B在QDQ图中是uint8，实际HMX使用signed byte权重加载。
context中存在 `convert_weights_to_signed.shuffled` 标记；该标记不是该准备函数
已经逐入口捕获的证据，具体转换与重排仍待独立探针。

每个模型只采样首个deep调用的实参，不宣称其全部调用配置均相同：

| 字段 | 五模型首个调用的值 | 静态用途 |
|---|---:|---|
| params[0] | 0 | 输出初始配置 |
| params[1] | 0x700 | 输出mxmem配置 |
| params[2] | 0 | activation初始配置 |
| params[3] | 0x71f | activation加载配置 |
| params[6] | 0x7ff | weight加载配置 |
| params[12] | 0x20 | 入口bit5测试选择deep分支 |
| conversion[0] | 1 | 转换描述符首字段 |

参数位域未全部解明，不能将0x71f简单叫作字节长度。实际首个activation/output
描述符保存在检查JSON中；此次没有读取完整原生A/W/O张量，没有建立逐元素布局。

## 探针如何证明路径

第一版 `identity` 对6个整数/FP16候选入口以及普通aligned主体加只读探针，
保存r0..r15、r31、predicates、USR，C helper遵循ABI；恢复寄存器并重放完整原packet。
每个重放packet的机器字节与原库对应packet逐字节相同。捕获真实返回PC和helper
返回PC，根据同进程的链接返回地址求模块基址，而不是按最近符号名猜调用者。
五模型最终计数探针均归一化到返回PC0x3a85cc，静态前一packet确实call 0x2a7d80。

初版1024³只收到82条入口与82条deep日志，不能将其除以4当调用次数。
计数版对每个候选执行原子累加，在Graph::execute返回后一次导出所有累积值。
四个Graph序号、返回值0、递增计数全部完整；普通aligned、unaligned、sparsity、
bnb候选和所监测FP16入口在该计数系列均为0。零计数只限定这几个探针点，
不等于已监测库中所有其他函数。

新增代码放ELF原先未映射的RX文件padding `0x993480`，计数版1681字节。
仅扩展首个RX LOAD的filesz/memsz，仍结束在下一RW LOAD `0x994000` 之前。
新增计数BSS位于 `0xa04000`，扩展原RW LOAD的memsz供loader零填充。
没有覆盖此前FP16探针使用的5×5内核或unaligned函数作为代码洞。
入口和Graph头部仍是此确切ELF的私有修改，不可移植到其他SDK地址。

## 数值检查与边界

全输出有限。QNN对CPU QDQ参考的量化整数网格索引最大差为1。
原版前、探针和恢复原版后的输出逐位一致，不代表QNN逐位等于CPU参考或FP32模型。

另一个直接输出观察是：两个1024³模型的全部输出精确等于整数网格值乘
FP16舍入后的输出scale；128³及M1则接近原FP32 scale网格，允许FP32算术舍入。
例如双输入1024³原scale为0.018872078508138657，FP16舍入后
0.0188751220703125。其最大CPU QDQ差0.019240379333496094略大于原一步长，
但整数网格差仍≤1，舍入scale网格残差为0。

首版检查只容许1.01步长而退出；第二版错误地将大形状的FP16-scale观察推广到
128³，也在数值检查退出。两个模型调用均成功，原始日志分别保留
`matmul_counts_attempt1.log`、`matmul_counts_attempt2.log`；最终检查明确区分
FP32/FP16 scale网格，并要求网格残差≤4个FP32 epsilon乘输出幅值上界。
实际反量化converter入口及其内部实现尚未捕获，不据输出规律归因某条指令。

独立 Hexagon 整数 HMX 最小例子已在设备上验证，见
[hmx_int8_minimal/README.md](examples/hmx_int8_minimal/README.md)。该文档记录了
自己的 FP32 反量化反汇编：`memub` → 减 zero point → `convert_w2sf` →
`sfmpy` → `memw`，以及它与 HMX `cvt.ub` 重定量化的区别。
这些是独立例子的实际指令，不能据此声称 QNN 反量化也使用相同函数或调度。

此次未确认完整DMA/HMX/HVX重叠、VTCM内部划分或整数原生布局；没有测试
A16W8、INT4、per-channel量化、所有形状或动态shape。
两个MatMul输入是运行时数据，**shape仍是静态的**。
只执行4次/进程，全部探针时间不用于速度排名；这不是INT8与FP16的正式性能对比。

## 精确身份与复现

- 原始QNN skel SHA256：`5d6900ed5c2edb2c81517adf251ca34a0eae43d41c7aeb799d855c66c84f0400`
- QNN runner SHA256：`fa346d116b79d7a917ade560e622d4261ad3ca00fdb359a26aeae2a61dea29b6`
- counts探针SHA256：`eda8f2fd10d4b2d05ec6f62e9b9c8053f3fac983b123a2a002393cbe2f0252d5`
- 模型与输入SHA：`models.json` / `models_matmul.json`；设备逐文件校验：
  `counts_device_hashes.txt` / `matmul_counts_device_hashes.txt`。

本机scratch目录 `.tmp/scratch/mnv2/qnn_int_matmul/`，手机私有目录
`/data/data/com.termux/files/home/csm/matmul_rebenchmark/qnn_int_matmul/`。
复用已有runner及原版baseline目录，不替换公共库。
每个模型调用前检查并等待llama-server、llama-cli、llama-bench，不终止用户任务。

```bash
cd /home/yydh/executorch
.tmp/locks/lockrun heavy -- env PYTHONPATH=/home/yydh QNN_SDK_ROOT=/home/yydh/env/qairt/2.41.0.251128 LD_LIBRARY_PATH=/home/yydh/env/qairt/2.41.0.251128/lib/x86_64-linux-clang TMPDIR=/home/yydh/executorch/.tmp/scratch/mnv2/qnn_int_matmul /home/yydh/miniconda3/envs/et/bin/python .tmp/scratch/mnv2/qnn_int_matmul/export_models.py
.tmp/locks/lockrun heavy -- env PYTHONPATH=/home/yydh QNN_SDK_ROOT=/home/yydh/env/qairt/2.41.0.251128 LD_LIBRARY_PATH=/home/yydh/env/qairt/2.41.0.251128/lib/x86_64-linux-clang TMPDIR=/home/yydh/executorch/.tmp/scratch/mnv2/qnn_int_matmul /home/yydh/miniconda3/envs/et/bin/python .tmp/scratch/mnv2/qnn_int_matmul/export_matmul.py
.tmp/locks/lockrun heavy -- env QINT_PROBE_MODE=counts TMPDIR=/home/yydh/executorch/.tmp/scratch/mnv2/qnn_int_matmul /home/yydh/miniconda3/envs/et/bin/python .tmp/scratch/mnv2/qnn_int_matmul/build_probe.py
.tmp/locks/lockrun device -- env QINT_PROBE_MODE=counts PYTHONPATH=/home/yydh /home/yydh/miniconda3/envs/et/bin/python .tmp/scratch/mnv2/qnn_int_matmul/run_device.py
.tmp/locks/lockrun device -- env QINT_CASES=models_matmul.json QINT_PROBE_MODE=counts PYTHONPATH=/home/yydh /home/yydh/miniconda3/envs/et/bin/python .tmp/scratch/mnv2/qnn_int_matmul/run_device.py
/home/yydh/miniconda3/envs/et/bin/python .tmp/scratch/mnv2/qnn_int_matmul/check_results.py
```

复现会覆盖本scratch同名结果与手机私有目录，需先另存历史记录。
`.tmp`脚本和模型不在Git内；跨机器需同步，不可只clone源码。
原始日志 `counts_<tag>.farf.log`，输出 `counts_{before/after}_<tag>.raw`、
`counts_<tag>.raw`，静态解码 `convbbb.asm`、`convbbb_deep.asm`、`caller.asm`，
探针源/汇编/manifest在 `counts/`。最终验收为 `dispatch_checks.json` 五条全部通过。
