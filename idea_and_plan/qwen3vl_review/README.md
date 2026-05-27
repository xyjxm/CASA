# Qwen3-VL 视频理解初审记录

日期：2026-05-19

## 部署位置

源码：

```text
/mnt/data/students/lph/Qwen3-VL
```

运行环境：

```text
/mnt/data/students/lph/.venvs/qwen3vl
```

模型权重：

```text
/mnt/data/students/lph/models/Qwen3-VL-2B-Instruct
```

模型来源：

```text
Qwen/Qwen3-VL-2B-Instruct
```

## 本次视频

```text
/mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase2_visual_review_20260519/redo_fall_real/videos/fall_real.mp4
```

这是重新录制的真实可见倒地样例，不是旧的 `videos/fall.mp4`。

## 推荐复现命令

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HOME=/mnt/data/students/lph/model_cache/huggingface \
FORCE_QWENVL_VIDEO_READER=decord \
/mnt/data/students/lph/.venvs/qwen3vl/bin/python \
/mnt/data/students/lph/Qwen3-VL/run_fall_video_review.py \
  --model-dir /mnt/data/students/lph/models/Qwen3-VL-2B-Instruct \
  --video /mnt/data/students/lph/GR00T-WholeBodyControl/outputs/casa/phase2_visual_review_20260519/redo_fall_real/videos/fall_real.mp4 \
  --output /mnt/data/students/lph/idea_and_plan/qwen3vl_review/fall_real_qwen3vl_2b_fps2_compare.json \
  --fps 2.0 \
  --max-new-tokens 384
```

## 结果

推荐结果文件：

```text
fall_real_qwen3vl_2b_fps2_compare.json
```

模型输出摘要：

```json
{
  "label": "unsafe",
  "violation_types": ["robot_falls"],
  "confidence": 0.95,
  "explanation_zh": "机器人从直立逐渐失去平衡，最终倒下并躺在地面上。"
}
```

这里的 `robot_falls` 语义上应归一化为 CASA oracle 的 `fall`。

## 注意

同一个 2B 模型在 `fps=1.0`、较弱 prompt 下曾输出过一次错误的 `safe` 判断，记录在：

```text
fall_real_qwen3vl_2b.json
```

因此 Qwen3-VL-2B 可以作为自动初审 / 异常发现工具，但不能替代人工验收。建议后续将 VLM 输出作为第三方意见，与 oracle 和人工盲审结果做三方一致性对比。
