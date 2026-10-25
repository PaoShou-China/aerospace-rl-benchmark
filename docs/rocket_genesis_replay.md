# 火箭 Genesis 轨迹采集与回放（首版）

本模块实现单副本真实回合采集、240 Hz 状态日志、接触/成功/结束事件、Genesis 双机位、同步 HUD、流式 MP4 和离线 HTML 分享。它独立于原来的 `scripts/visualize.py`，只给火箭环境加入默认关闭的 `visual_recorder` 观察口。没有修改动力学、奖励、动作映射或成功门槛。

## 运行条件与验证边界

- 真实采集需要仓库 `requirements.txt` 中的 PyTorch、Genesis 等依赖。优先使用仓库已验证的 Python 3.10 / Linux 环境和固定版本 `genesis-world==1.4.4`。
- CPU 单副本可用于诊断；GPU 需要 CUDA 设备与对应 PyTorch。离屏相机还需要可用图形驱动/渲染环境，不能只检查 `torch.cuda.is_available()`。
- 视频需要 imageio、imageio-ffmpeg（已在根 requirements 中）。HTML 文件内嵌视频与遥测，不依赖网络，大小由视频决定。
- 本次开发环境没有 PyTorch、Genesis 或 NVIDIA 设备。因此已验证的是 Python 语法、日志/事件/指标/插值及 HUD 数据链；**真实 Genesis 采集、相机 API 与图形环境仍需在组内机器上进行集成验证**。不能把单元测试夹具当作真实着陆结果。

从仓库根目录运行以下命令。

## 1. 采集一条真实轨迹

```bash
python scripts/rocket_replay.py collect --backend cpu --seed 7 --out results/rocket_replay/seed7
```

默认使用 PDG 基线、一个副本和默认静风环境。它不会悄悄读取旧训练目录。若需要某次训练的风扰/奖励等配置，显式传入：

```bash
python scripts/rocket_replay.py collect --backend gpu --seed 13 --config results/rocket/config.json --out results/rocket_replay/seed13
python scripts/rocket_replay.py collect --backend gpu --seed 21 --config results/rocket/config.json --checkpoint results/rocket/checkpoint.pt --out results/rocket_replay/policy21
```

每次只保存第一个回合，不使用“总结束回合数达到副本数”的批量停止条件。换 seed 收集真实成功和失败，结果由环境原始标记给出，不保证指定 seed 一定成功。输出目录已存在时拒绝覆盖。

`--max-steps` 只是采集上限，提前停止标为 `collection_limited`，不会伪造环境超时。`--terminal-height` 默认 20 m，是“下降足端首次进入该高度”的显示分析定义，不改变任务。

## 2. 检查数据（不需要 Genesis）

```bash
python scripts/rocket_replay.py inspect --episode results/rocket_replay/seed7 --time 10
```

输出：`manifest.json`、`trajectory.json`、`states.npz`、`events.jsonl`、`policy_steps.json`、`summary.json`。JSON 使用 `aerospace-visual-state/v1`、局部 ENU、wxyz 机体到世界四元数、SI 单位。NPZ 无需 pickle。

- 日志起点是 reset/warmup **之后**，时间轴不计入 warmup。
- 每个实际物理子步采样。终止后冻结的控制周期与下一回合不混入。
- `pad_contact` / `ground_contact` 为首次检测事件；`pad_contact_gate` 另存环境判据使用的控制周期起点速度模长与角速度模长。检测时间精度为 1/240 s，不声称得到连续时间精确碰撞时刻。
- `settled` 与 `terminated` / `time_limit` 分开记录。成功后仍按环境运行到结束；`complete` 与 `terminated` 可同时存在，日志保留原始结果。
- 燃料比例是满箱比例；质量来自实际比例 × 容量，消耗百分比相对本回合实际初始燃料。默认满箱 410.9 kg，初始 5% 约 20.545 kg。
- 足端由 URDF 三个着陆腿的碰撞盒角点计算。正立基座参考高度约 2.575 m，仅用于显示误差；不代替成功判据。倾覆时足端间隙不代表全机体碰撞距离。

## 3. Genesis 双机位 MP4

```bash
python scripts/rocket_replay.py render --episode results/rocket_replay/seed7 --out results/rocket_replay/seed7/full.mp4
python scripts/rocket_replay.py render --episode results/rocket_replay/seed7 --out results/rocket_replay/seed7/slow.mp4 --start 9 --end 12 --speed 0.25 --fps 60
```

慢放区间必须处于实际轨迹范围，先用日志查接触时刻。默认 1280×720、30 fps；最小 960×720，宽高需为偶数。`--font` 可指定本地字体。

显示场景只设置位姿，不调用 `scene.step()`。两台相机在同一时刻渲染：全程追踪与台面固定近景。HUD 包含位置、速度、位置偏差、足端间隙、燃料消耗与触地/末端阶段计时。按实际时间戳重采样，四元数 SLERP、状态标记零阶保持。视频逐帧编码，不积累整段 RGB。

背景提供地面、台面标记、道路和少量建筑。显示 URDF 清除固定的常亮 flame，另用预分配几何体按实际油门/喷管方向显示简化喷流。栅格舵实际状态已记录，但舵面独立动画、扬尘、高级天空、自由交互播放器和外部 FlightGear/Vizard 适配尚未实现，不能作为本版已完成项。

回放核对采集时的资产哈希，不匹配会拒绝渲染。先使用相同资产版本，不能为了好看改碰撞或缩放后继续宣称同一接触几何。

## 4. 可转发的离线 HTML

```bash
python scripts/rocket_replay.py html --episode results/rocket_replay/seed7 --video results/rocket_replay/seed7/full.mp4 --out results/rocket_replay/seed7/share.html
```

渲染自动产生 `.timeline.json`，分享时校验其轨迹指纹。HTML 内嵌 Genesis 视频及与媒体时间对应的真实遥测，支持浏览器播放、暂停和时间拖动；它不是在浏览器里运行 Genesis，也不提供任意三维视角。保留视频旁的 timeline 文件，否则无法导出。

## 检查与集成验收

```bash
python -m unittest discover -s tests -p 'test_rocket_replay.py' -v
python -m compileall -q visualization scripts/rocket_replay.py environments/rocket/env.py
```

数据测试只用 NumPy/Pillow。组内还需验证：同 seed 同设备记录关闭/开启的动作和状态一致；真实回合终止末帧没有重置瞬移；240 Hz 接触时刻和 120 Hz 判据速度对应；正常速度时长与日志相符；双机位同步；完整 30 s 输出内存保持有界；HTML 断网仍可打开。

本次不修改原有两个通用评测脚本的回合统计问题，避免将视景改动与 benchmark 评测语义变更混在一起。
