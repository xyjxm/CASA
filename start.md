这个问题表面症状是机器人在 policy 启动后仍然频繁摔倒，甚至出现 MuJoCo 数值不稳定 warning；最终根因不是 policy 推理慢，而是 deployment 端的线程绑核和 lowcmd DDS 发布频率把 Python MuJoCo sim 的 200 Hz 主循环拖慢了。
效果和代码
暂时无法在飞书文档外展示此内容
暂时无法在飞书文档外展示此内容
启动命令
终端 1：MuJoCo sim
cd /mnt/data/students/lph/GR00T-WholeBodyControl
source .venv_sim/bin/activate

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
taskset -c 0-5 \
python -u gear_sonic/scripts/run_sim_loop.py \
  --no-enable-onscreen \
  --enable-offscreen \
  --enable-image-publish \
  --camera-port 5555 \
  --image-publish-fps 10 \
  --offscreen-camera-width 480 \
  --offscreen-camera-height 360 \
  --drop-on-start \
  --sim-frequency 200 \
  --fall-log-interval-seconds 30 \
  --fall-stats \
  --sim-timing-log-interval-seconds 5

终端 2：C++ deployment
cd /mnt/data/students/lph/GR00T-WholeBodyControl/gear_sonic_deploy
source ../scripts/setup_no_root_env.sh

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 6-15 \
bash deploy.sh --input-type keyboard sim \
  --quiet \
  --timing-log-interval-seconds 10 \
  --command-publish-frequency 100

终端 3 ：OpenCV 可视化
cd /mnt/data/students/lph/GR00T-WholeBodyControl
source .venv_sim/bin/activate

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 16-18 \
python -u gear_sonic/scripts/run_camera_viewer.py \
  --camera-host localhost \
  --camera-port 5555 \
  --fps 10 \
  --max-display-width 480

操作按键
deployment 初始化完成后：
]   start policy
T   play motion
O   stop
背景
目标运行方式：
- MuJoCo sim 端用 Python 启动。
- C++ deployment 端加载 TensorRT policy / encoder / planner。
- 为了复现 viewer 中按 9 解开弹性绳，sim 端支持 --drop-on-start。
最开始看到的现象是：
Warning: Robot has fallen, height: 0.196 m (falls=13, elapsed=27.4s)
Warning: Robot has fallen, height: 0.199 m (falls=18, elapsed=37.5s)
WARNING: Nan, Inf or huge value in QACC at DOF 3. The simulation is unstable.
policy 启动前松绳后摔倒并不是关键，因为 sim 会 reset，机器人会重新回到正常站立状态。真正的问题是 policy 启动后 sim / control 时序仍然不够实时，导致机器人持续摔倒。
先排除的方向
日志
第一步减少了高频 console 输出：
- sim 端 Robot has fallen 限频打印。
- sim 退出时打印总摔倒次数和 falls/min。
- deployment 端加 --quiet，关闭 Loop timing 等周期性日志。
减少日志后摔倒频率下降，说明 console I/O 对实时性有影响，但它不是唯一根因。
可视化
随后排除了可视化路径：
python gear_sonic/scripts/run_sim_loop.py \
  --no-enable-onscreen \
  --drop-on-start \
  --fall-log-interval-seconds 30 \
  --fall-stats
这个命令没有：
- onscreen MuJoCo viewer
- offscreen render
- image publish
- OpenCV viewer
- X11 转发
- JPEG / ZMQ 图像发送
但机器人仍然频繁摔倒。因此问题不在可视化。
100 Hz sim frequency
曾尝试把 sim 从 200 Hz 降到 100 Hz 来减轻 CPU 压力，但这不是好方向。100 Hz 会把 MuJoCo timestep 从 0.005s 放大到 0.01s，对高增益 PD 控制和接触丰富的人形机器人更容易导致数值不稳定：
WARNING: Nan, Inf or huge value in QACC
因此 sim 频率应该保留 200 Hz，优化 wall-clock 实时性，而不是降低 MuJoCo 物理频率。
增加实时性测量
为避免继续猜测，sim 端增加了：
--sim-timing-log-interval-seconds 5
它会周期性打印：
Sim timing: target_hz=200.0, actual_hz=195.0, avg_step=3.50ms, max_step=8.26ms, overruns=13/975 (1.3%)
字段含义：
字段
含义
target_hz
目标 sim 频率，默认 200 Hz
actual_hz
实际 wall-clock 频率
avg_step
单步平均耗时，不含 sleep
max_step
统计窗口内最大单步耗时
overruns
单步耗时超过目标 timestep 的次数
判断标准：
- actual_hz 接近 195-200：基本健康。
- actual_hz 长期低于 190：sim 跑不满实时。
- overruns 长期超过 10%：调度或通信压力偏大。
- max_step 出现几十毫秒：有明显卡顿，会破坏控制时序。
deployment 端使用：
--timing-log-interval-seconds 1
观察：
Loop timing - LowState age: 2.749ms,
Obs: 396us,
Policy: 99us,
Obs 2 Motor Command: 495us,
Post processing: 44us
这里可以判断 C++ policy/control 是否慢。实测 Obs 2 Motor Command 只有约 0.4-0.7ms，远小于 20 ms 控制周期，因此 C++ policy 推理不是瓶颈。
关键实验结果
sim 单独运行
只运行 Python MuJoCo sim：
actual_hz ~= 195
avg_step ~= 3.4-3.5ms
overruns ~= 0-2%
说明 Python MuJoCo 本身在这台机器上基本可以跑满 200 Hz。
deployment 启动后，未修复前
启动 C++ deployment 后，sim timing 明显变差：
actual_hz ~= 120-145
overruns ~= 50%
这个量级的掉帧足以解释机器人持续摔倒。policy 端虽然很快，但 sim 物理主循环已经不能按 200 Hz 更新，DDS 状态和控制命令时序都会抖。
根因 1：deployment 主线程被硬绑到 CPU 0
即使用外部命令把 sim 和 deployment 分开：
taskset -c 0-5   # sim
taskset -c 6-15  # deployment
仍然发现 deployment 主线程跑在 CPU 0。原因是 C++ 代码中有硬编码：
CPU_SET(0, &cpuset);
pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
这会覆盖外部 taskset，导致 deployment 主线程和 sim 抢 CPU 0。
修复方式：
- 默认不再硬绑 CPU 0。
- 尊重外部 taskset。
- 如果确实需要手动指定 deployment 主线程 CPU，可以使用环境变量：
G1_DEPLOY_MAIN_CPU=12
修复后检查 affinity：
taskset -pc <sim_pid>
taskset -pc <deploy_pid>
期望：
sim:    0-5
deploy: 6-15
根因 2：500 Hz lowcmd DDS 发布压垮 sim
deployment 原始 lowcmd 发布频率是 500 Hz：
publish_dt_ = 0.002;
对真实机器人这通常合理，但对这个 Python MuJoCo sim2sim 路径太重。sim 端有 DDS 接收线程 recvUC，deployment 端有 command_writer 线程。实测中这些线程会持续占用 CPU，导致 sim 主循环变慢。
因此增加了参数：
--command-publish-frequency HZ
测试结果：
command publish frequency
sim actual_hz
现象
500 Hz
120-145 Hz
严重不实时，频繁摔倒
200 Hz
178-186 Hz
明显改善，但仍偏慢
100 Hz
188-193 Hz
当前机器上较稳
由于 C++ control loop 本身是 50 Hz，sim2sim 中用 100 Hz 发布 lowcmd 通常足够，比 500 Hz 更适合这台机器。

