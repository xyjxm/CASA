"""JSONL worker for real NaVid/Uni-NaVid model inference.

This module is intentionally dependency-isolated. The SONIC runner can execute
it with a NaVid-specific Python environment while keeping the simulator
environment unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
from PIL import Image


def _write_json(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


def _ensure_vision_tower_link(repo_path: Path, vision_tower_path: Path | None) -> Path | None:
    if vision_tower_path is None:
        return None
    target = vision_tower_path.resolve()
    if not target.exists():
        raise FileNotFoundError(f"vision tower does not exist: {target}")
    model_zoo = repo_path / "model_zoo"
    model_zoo.mkdir(parents=True, exist_ok=True)
    link_path = model_zoo / "eva_vit_g.pth"
    if link_path.exists():
        return link_path
    link_path.symlink_to(target)
    return link_path


class RealNaVidModel:
    def __init__(
        self,
        *,
        repo_path: Path,
        model_path: Path,
        variant: str,
        vision_tower_path: Path | None,
        max_new_tokens: int,
        temperature: float,
    ) -> None:
        self.repo_path = repo_path.resolve()
        self.model_path = model_path.resolve()
        self.variant = variant
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.history_rgb_tensor = None
        self.rgb_list: list[np.ndarray] = []
        self.conv_mode = "vicuna_v1"

        if variant not in {"navid", "uni-navid"}:
            raise ValueError("variant must be navid or uni-navid")
        if not self.repo_path.exists():
            raise FileNotFoundError(f"repo path does not exist: {self.repo_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"model path does not exist: {self.model_path}")

        _ensure_vision_tower_link(self.repo_path, vision_tower_path)
        os.chdir(self.repo_path)
        sys.path.insert(0, str(self.repo_path))

        if variant == "navid":
            from navid.constants import (  # type: ignore
                DEFAULT_IMAGE_TOKEN,
                IMAGE_TOKEN_INDEX,
            )
            from navid.conversation import SeparatorStyle, conv_templates  # type: ignore
            from navid.mm_utils import (  # type: ignore
                KeywordsStoppingCriteria,
                get_model_name_from_path,
                tokenizer_image_token,
            )
            from navid.model.builder import load_pretrained_model  # type: ignore
        else:
            from uninavid.constants import (  # type: ignore
                DEFAULT_IMAGE_TOKEN,
                IMAGE_TOKEN_INDEX,
            )
            from uninavid.conversation import SeparatorStyle, conv_templates  # type: ignore
            from uninavid.mm_utils import (  # type: ignore
                KeywordsStoppingCriteria,
                get_model_name_from_path,
                tokenizer_image_token,
            )
            from uninavid.model.builder import load_pretrained_model  # type: ignore

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for strict NaVid real-model inference")

        self.torch = torch
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.SeparatorStyle = SeparatorStyle
        self.conv_templates = conv_templates
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.tokenizer_image_token = tokenizer_image_token

        model_name = get_model_name_from_path(str(self.model_path))
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            str(self.model_path),
            None,
            model_name,
            device_map="auto",
            device="cuda",
        )
        self.model.eval()
        self.model_name = model_name
        self.prompt_template = (
            "Imagine you are a robot programmed for navigation tasks. You have been given a "
            "video of historical observations and an image of the current observation <image>. "
            "Your assigned task is: '{}'. Analyze this series of images to decide your next "
            "move, which could involve turning left or right by a specific degree or moving "
            "forward a certain distance."
        )

    def reset(self) -> None:
        self.history_rgb_tensor = None
        self.rgb_list = []

    @property
    def device(self):
        return getattr(self.model, "device", self.torch.device("cuda:0"))

    def _load_images(self, image_paths: list[str]) -> list[np.ndarray]:
        images: list[np.ndarray] = []
        for image_path in image_paths:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"image path does not exist: {path}")
            with Image.open(path) as image:
                images.append(np.asarray(image.convert("RGB")))
        return images

    def _process_images(self) -> list[Any]:
        start_img_index = 0
        if self.history_rgb_tensor is not None:
            start_img_index = self.history_rgb_tensor.shape[0]
        batch = np.asarray(self.rgb_list[start_img_index:])
        if batch.size == 0:
            if self.history_rgb_tensor is None:
                raise ValueError("no RGB frames available for NaVid inference")
            return [self.history_rgb_tensor]

        video = self.image_processor.preprocess(batch, return_tensors="pt")["pixel_values"]
        video = video.to(device=self.device, dtype=self.torch.float16)
        if self.history_rgb_tensor is None:
            self.history_rgb_tensor = video
        else:
            self.history_rgb_tensor = self.torch.cat((self.history_rgb_tensor, video), dim=0)
        return [self.history_rgb_tensor]

    def predict(self, *, instruction: str, image_paths: list[str]) -> str:
        if not image_paths:
            raise ValueError("real NaVid inference requires at least one image path")
        self.rgb_list.extend(self._load_images(image_paths))
        prompt = self.prompt_template.format(instruction)

        torch = self.torch
        question = prompt.replace(self.DEFAULT_IMAGE_TOKEN, "").replace("\n", "")
        qs = prompt

        video_start = self.tokenizer("<video_special>", return_tensors="pt").input_ids[0][1:].to(self.device)
        video_end = self.tokenizer("</video_special>", return_tensors="pt").input_ids[0][1:].to(self.device)
        image_start = self.tokenizer("<image_special>", return_tensors="pt").input_ids[0][1:].to(self.device)
        image_end = self.tokenizer("</image_special>", return_tensors="pt").input_ids[0][1:].to(self.device)
        navigation = self.tokenizer("[Navigation]", return_tensors="pt").input_ids[0][1:].to(self.device)
        image_separator = self.tokenizer("<image_sep>", return_tensors="pt").input_ids[0][1:].to(self.device)

        if self.model.config.mm_use_im_start_end:
            qs = "<im_start><image><im_end>\n" + qs.replace("<image>", "")
        else:
            qs = self.DEFAULT_IMAGE_TOKEN + "\n" + qs.replace("<image>", "")

        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        conv_prompt = conv.get_prompt()

        token_prompt = self.tokenizer_image_token(
            conv_prompt,
            self.tokenizer,
            self.IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).to(self.device)
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start)
            new_list.append(image_separator)
            new_list.append(token_prompt[idx : idx + 1])
            new_list.append(video_end)
            new_list.append(image_start)
            new_list.append(image_end)
            new_list.append(navigation)
            token_prompt = token_prompt[idx + 1 :]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        stopping_criteria = self.KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
        images = self._process_images()

        with torch.inference_mode():
            self.model.update_prompt([[question]])
            output_ids = self.model.generate(
                input_ids,
                images=images,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        return outputs.strip()


def _handle_loop(model: RealNaVidModel) -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "reset":
                model.reset()
                _write_json({"ok": True, "type": "reset"})
            elif command == "predict":
                output = model.predict(
                    instruction=str(request["instruction"]),
                    image_paths=[str(path) for path in request.get("image_paths", [])],
                )
                _write_json(
                    {
                        "ok": True,
                        "type": "prediction",
                        "request_id": request.get("request_id"),
                        "raw_output": output,
                    }
                )
            elif command == "health":
                _write_json({"ok": True, "type": "health", "model_loaded": True})
            elif command == "close":
                _write_json({"ok": True, "type": "close"})
                return
            else:
                raise ValueError(f"unknown command: {command}")
        except Exception as exc:  # noqa: BLE001 - worker must return structured errors.
            _write_json(
                {
                    "ok": False,
                    "type": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--variant", choices=["navid", "uni-navid"], default="navid")
    parser.add_argument("--vision-tower-path", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        model = RealNaVidModel(
            repo_path=args.repo_path,
            model_path=args.model_path,
            variant=args.variant,
            vision_tower_path=args.vision_tower_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        _write_json(
            {
                "ok": True,
                "type": "ready",
                "model_loaded": True,
                "variant": args.variant,
                "model_path": str(args.model_path),
                "model_name": model.model_name,
                "context_len": model.context_len,
                "device": str(model.device),
            }
        )
        _handle_loop(model)
        return 0
    except Exception as exc:  # noqa: BLE001 - startup errors are part of probe output.
        _write_json(
            {
                "ok": False,
                "type": "ready",
                "model_loaded": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
