#!/usr/bin/env python3
"""
MotusWanVlmDirectMask Inference API Server

Provides REST endpoints for real-world Motus inference using MotusWanVlmDirectMask model.
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
from transformers import AutoProcessor

PROJ_ROOT = str(Path(__file__).resolve().parents[3])
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from models.motus_wan_vlm_direct_mask import MotusWanVlmDirectMask, MotusWanVlmDirectMaskConfig
from wan.modules.t5 import T5EncoderModel
from data.utils.image_utils import resize_with_padding


log = logging.getLogger("motus_api_server")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


app = FastAPI(
    title="MotusWanVlmDirectMask Inference API",
    description="REST API for MotusWanVlmDirectMask model inference",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferenceRequest(BaseModel):
    instruction: Optional[str] = Field(default=None, description="Task instruction. Falls back to server default when omitted.")
    images: Optional[List[str]] = Field(default=None, description="List of base64 encoded images.")
    proprio_data: Optional[List[List[float]]] = Field(
        default=None,
        description="Custom proprioception sequence. The last state is used for inference.",
    )
    image: Optional[str] = Field(default=None, description="Single base64 encoded RGB image.")
    image_path: Optional[str] = Field(default=None, description="Local path to RGB image.")
    state: Optional[List[float]] = Field(default=None, description="Backward-compatible single robot state vector.")
    t5_embeddings_path: Optional[str] = Field(default=None, description="Path to T5 embedding .pt file.")
    auto_find_t5_embeddings: bool = Field(default=True)
    num_inference_steps: Optional[int] = Field(default=None, ge=1)
    return_frame_grid: bool = Field(default=False)
    save_output_path: Optional[str] = Field(default=None)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str


class InferenceResponse(BaseModel):
    predicted_actions: List[List[float]]
    instruction: str
    action_head_type: str
    processing_time_ms: float
    model_info: Dict[str, Any]
    timestamp: str
    action_shape: Optional[List[int]] = None
    predicted_frames_shape: Optional[List[int]] = None
    frame_grid_image: Optional[str] = None


class ServerState:
    def __init__(self) -> None:
        self.model: Optional[MotusWanVlmDirectMask] = None
        self.processor: Optional[AutoProcessor] = None
        self.t5_encoder: Optional[T5EncoderModel] = None
        self.config_dict: Optional[Dict[str, Any]] = None
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_steps: int = 10
        self.lock = threading.Lock()
        self.checkpoint_path: Optional[str] = None
        self.wan_path: Optional[str] = None
        self.vlm_path: Optional[str] = None
        self.t5_embeddings_dir: Optional[str] = None
        self.default_instruction: Optional[str] = None
        self.default_t5_embeddings_path: Optional[str] = None
        self.default_language_embeddings: Optional[List[torch.Tensor]] = None
        self.action_min: Optional[np.ndarray] = None
        self.action_max: Optional[np.ndarray] = None
        self.action_denorm_required: bool = False


SERVER_STATE = ServerState()
DEFAULT_SCENE_PREFIX = (
    "The whole scene is in a realistic environment. "
    "The robot is currently performing the following task: "
)

# Pre-computed action normalization stats for different datasets
DATASET_ACTION_STATS = {
    "dobot_pour_water": {
        "min": [-2.5780635, -0.8699569, -2.4356816, -0.85862976, 1.2083772, -0.35320008, 0.0, 0.8591288, -0.5598094, 0.62409264, -1.0244704, -2.1347938, -3.0375712, 0.0],
        "max": [-0.7302208, 0.660163, -0.30455458, 1.2712903, 2.1897526, 4.2206035, 0.9999596, 2.2337525, 1.0345864, 2.4839027, 0.55300254, -1.3848281, -1.0665554, 0.9998678],
    },
    "dobot_cook_vegetable": {
        "min": [-2.067998, -0.80020654, -2.4999998, -0.4177313, 1.1323755, 0.93191046, 0.10638021, 1.0747652, -0.39955214, 0.71550626, -1.5966251, -2.01556, -3.5221725, 0.0],
        "max": [-1.1761625, 0.46642998, -1.1376227, 1.7536696, 2.0811172, 2.9992201, 1.0, 1.9841311, 1.0186752, 2.3929965, 0.5782303, -1.1281374, -0.18062241, 0.99811196],
    },
}


def load_yaml_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def swap_rgb_bgr(image: Image.Image) -> Image.Image:
    # Match training data where BGR video frames were interpreted as RGB.
    return Image.fromarray(np.array(image.convert("RGB"))[:, :, ::-1])


def decode_base64_image(image_base64: str) -> Image.Image:
    if image_base64.startswith("data:image"):
        image_base64 = image_base64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_base64)
    return swap_rgb_bgr(Image.open(io.BytesIO(image_bytes)))


def resize_image_with_padding(image: Image.Image, size_hw: tuple[int, int]) -> Image.Image:
    image_np = np.array(image).astype(np.uint8)
    resized_np = resize_with_padding(image_np, size_hw)
    return Image.fromarray(resized_np)


def image_to_tensor(image: Image.Image, size_hw: tuple[int, int]) -> torch.Tensor:
    image = resize_image_with_padding(image, size_hw)
    image_np = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)


def compose_multiview_image(images: List[Image.Image], size_hw: Optional[tuple[int, int]] = None) -> Image.Image:
    """Compose multiple images into the Dobot T-shape layout used by the converter."""
    if len(images) == 1:
        return images[0].convert("RGB")

    if size_hw is not None:
        target_h, target_w = size_hw
        top_h = max(1, target_h * 2 // 3)
        bottom_h = max(1, target_h - top_h)
        left_w = max(1, target_w // 2)
        right_w = max(1, target_w - left_w)
        top_w = target_w
    else:
        top = np.array(images[0].convert("RGB")).astype(np.uint8)
        top_h, top_w = top.shape[:2]
        bottom_h = max(1, top_h // 2)
        left_w = max(1, top_w // 2)
        right_w = max(1, top_w - left_w)

    if len(images) == 2:
        left_img = np.array(images[1].convert("RGB")).astype(np.uint8)
        right_img = left_img
    else:
        left_img = np.array(images[1].convert("RGB")).astype(np.uint8)
        right_img = np.array(images[2].convert("RGB")).astype(np.uint8)

    top = np.array(Image.fromarray(np.array(images[0].convert("RGB")).astype(np.uint8)).resize((top_w, top_h), Image.BICUBIC))
    left_resized = np.array(Image.fromarray(left_img).resize((left_w, bottom_h), Image.BICUBIC))
    right_resized = np.array(Image.fromarray(right_img).resize((right_w, bottom_h), Image.BICUBIC))
    bottom_row = np.concatenate([left_resized, right_resized], axis=1)
    composed = np.concatenate([top, bottom_row], axis=0)
    return Image.fromarray(composed)


def load_default_instruction(default_instruction: Optional[str], default_instruction_file: Optional[str]) -> Optional[str]:
    if default_instruction_file:
        text = Path(default_instruction_file).expanduser().read_text(encoding="utf-8").strip()
        return text or None
    if default_instruction and default_instruction.strip():
        return default_instruction.strip()
    return None


def build_vlm_inputs(
    processor: AutoProcessor,
    instruction: str,
    image: Image.Image,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    # Create VLM messages format - MATCH dataset order: image first, text second
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    # Apply chat template with add_generation_prompt=True (MATCH dataset behavior)
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # Process vision info using qwen_vl_utils (MATCH dataset approach)
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)

    # Get final processor inputs (MATCH dataset approach)
    encoded = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    vlm_inputs: Dict[str, torch.Tensor] = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "pixel_values": encoded["pixel_values"].to(device),
    }
    if encoded.get("image_grid_thw") is not None:
        vlm_inputs["image_grid_thw"] = encoded["image_grid_thw"].to(device)
    return vlm_inputs


def resolve_wan_model_dir(path: str) -> str:
    raw_path = Path(path).expanduser().resolve()
    if (raw_path / "Wan2.2_VAE.pth").exists():
        return str(raw_path)
    nested_path = raw_path / "Wan2.2-TI2V-5B"
    if (nested_path / "Wan2.2_VAE.pth").exists():
        return str(nested_path)
    raise ValueError(f"Cannot resolve WAN model directory from {path}")


def load_server_components(
    model_config_path: str,
    ckpt_dir: str,
    wan_path: str,
    vlm_path: str,
    device: Optional[str],
    t5_embeddings_dir: Optional[str],
    default_instruction: Optional[str],
    default_instruction_file: Optional[str],
    default_t5_embeddings_path: Optional[str],
    action_min: Optional[np.ndarray] = None,
    action_max: Optional[np.ndarray] = None,
) -> None:
    resolved_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    config_dict = load_yaml_config(model_config_path)

    wan_path = resolve_wan_model_dir(wan_path)
    vlm_path = str(Path(vlm_path).expanduser().resolve()) if vlm_path else config_dict["model"]["vlm"]["checkpoint_path"]

    # Create MotusWanVlmDirectMaskConfig
    common = config_dict["common"]
    model_cfg = config_dict["model"]

    # Fix: YAML loads 1e-5 as string, need to convert to float
    def to_float(val):
        if isinstance(val, str):
            return float(val)
        return val

    model_config = MotusWanVlmDirectMaskConfig(
        wan_checkpoint_path=wan_path,
        vae_path=os.path.join(wan_path, "Wan2.2_VAE.pth"),
        wan_config_path=wan_path,
        vlm_checkpoint_path=vlm_path,
        video_precision="bfloat16",
        action_state_dim=common["state_dim"],
        action_dim=common["action_dim"],
        action_expert_dim=model_cfg["action_expert"]["hidden_size"],
        action_expert_ffn_dim_multiplier=model_cfg["action_expert"]["ffn_dim_multiplier"],
        action_expert_norm_eps=to_float(model_cfg["action_expert"].get("norm_eps", 1e-6)),
        vlm_dim=model_cfg["qwen3_expert"]["vlm_dim"],
        qwen3_expert_head_dim=model_cfg["qwen3_expert"]["head_dim"],
        qwen3_expert_num_heads=model_cfg["qwen3_expert"]["num_heads"],
        qwen3_expert_num_layers=model_cfg["qwen3_expert"]["num_layers"],
        qwen3_expert_norm_eps=to_float(model_cfg["qwen3_expert"].get("norm_eps", 1e-5)),
        global_downsample_rate=common["global_downsample_rate"],
        video_action_freq_ratio=common["video_action_freq_ratio"],
        num_video_frames=common["num_video_frames"],
        video_height=common["video_height"],
        video_width=common["video_width"],
        batch_size=1,
        video_loss_weight=model_cfg["loss_weights"]["video_loss_weight"],
        action_loss_weight=model_cfg["loss_weights"]["action_loss_weight"],
        training_mode="finetune",
        load_pretrained_backbones=False,
        vlm_frozen=getattr(model_cfg["vlm"], "frozen", False),
    )

    # Create model
    model = MotusWanVlmDirectMask(model_config).to(resolved_device)
    model.load_checkpoint(ckpt_dir, strict=False)
    model.eval()
    # Fix: VLM was hardcoded to cuda:0 in model, move it to correct device
    model.vlm_model = model.vlm_model.to(resolved_device)
    log.info(f"Loaded MotusWanVlmDirectMask from {ckpt_dir}")

    # Load VLM processor
    processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
    log.info(f"Loaded VLM processor from {vlm_path}")

    SERVER_STATE.model = model
    SERVER_STATE.processor = processor
    SERVER_STATE.config_dict = config_dict
    SERVER_STATE.device = resolved_device
    SERVER_STATE.default_steps = int(model_cfg.get("inference", {}).get("num_inference_timesteps", 10))
    SERVER_STATE.checkpoint_path = ckpt_dir
    SERVER_STATE.wan_path = wan_path
    SERVER_STATE.vlm_path = vlm_path
    SERVER_STATE.t5_embeddings_dir = t5_embeddings_dir
    SERVER_STATE.default_instruction = load_default_instruction(default_instruction, default_instruction_file)
    SERVER_STATE.default_t5_embeddings_path = (
        str(Path(default_t5_embeddings_path).expanduser().resolve())
        if default_t5_embeddings_path
        else None
    )
    SERVER_STATE.default_language_embeddings = None
    SERVER_STATE.action_min = action_min
    SERVER_STATE.action_max = action_max
    SERVER_STATE.action_denorm_required = (action_min is not None and action_max is not None)

    if SERVER_STATE.default_t5_embeddings_path is not None:
        SERVER_STATE.default_language_embeddings = load_t5_embeddings(
            SERVER_STATE.default_t5_embeddings_path,
            torch.device("cpu"),
        )
        log.info("Loaded default T5 embeddings from %s", SERVER_STATE.default_t5_embeddings_path)


def get_input_image(request: InferenceRequest, size_hw: tuple[int, int]) -> Image.Image:
    if request.images is not None:
        if len(request.images) == 0:
            raise ValueError("images cannot be an empty list.")
        decoded_images = [decode_base64_image(image_b64) for image_b64 in request.images]
        return compose_multiview_image(decoded_images, size_hw=size_hw)
    if request.image is not None:
        return decode_base64_image(request.image)
    if request.image_path is not None:
        return swap_rgb_bgr(Image.open(request.image_path))
    raise ValueError("Provide image or images")


def resolve_instruction(request: InferenceRequest) -> str:
    if request.instruction is not None and request.instruction.strip():
        return request.instruction.strip()
    if SERVER_STATE.default_instruction is not None:
        return SERVER_STATE.default_instruction
    raise ValueError("No instruction provided and no default instruction configured on the server.")


def resolve_state_values(request: InferenceRequest) -> Optional[List[float]]:
    if request.state is not None:
        return request.state
    if request.proprio_data is not None:
        if len(request.proprio_data) == 0:
            raise ValueError("proprio_data cannot be an empty list.")
        return [float(x) for x in request.proprio_data[-1]]
    return None


def get_state_tensor(state_values: Optional[List[float]], state_dim: int, device: torch.device) -> torch.Tensor:
    if state_values is None:
        return torch.zeros((1, state_dim), dtype=torch.float32, device=device)

    if len(state_values) != state_dim:
        raise ValueError(f"State length mismatch: expected {state_dim}, got {len(state_values)}.")

    # No normalization — pass raw qpos directly to model
    return torch.tensor(state_values, dtype=torch.float32, device=device).unsqueeze(0)


def load_t5_embeddings(path: str, device: torch.device) -> List[torch.Tensor]:
    loaded = torch.load(path, map_location=device)
    if isinstance(loaded, torch.Tensor):
        return [loaded.to(device)]
    if isinstance(loaded, list):
        return [t.to(device) for t in loaded]
    raise ValueError("Unsupported T5 embedding format")


def resolve_language_embeddings(request: InferenceRequest, instruction: str) -> List[torch.Tensor]:
    if request.t5_embeddings_path:
        return load_t5_embeddings(request.t5_embeddings_path, SERVER_STATE.device)

    if request.auto_find_t5_embeddings and SERVER_STATE.t5_embeddings_dir:
        import hashlib

        slug = hashlib.md5(instruction.encode()).hexdigest()
        t5_path = Path(SERVER_STATE.t5_embeddings_dir) / f"{slug}.pt"
        if t5_path.exists():
            log.info("Auto-loaded T5 from %s", t5_path)
            return load_t5_embeddings(str(t5_path), SERVER_STATE.device)

    if SERVER_STATE.default_language_embeddings is not None:
        return [embedding.to(SERVER_STATE.device) for embedding in SERVER_STATE.default_language_embeddings]

    if SERVER_STATE.default_t5_embeddings_path is not None:
        return load_t5_embeddings(SERVER_STATE.default_t5_embeddings_path, SERVER_STATE.device)

    raise ValueError(
        "No T5 embeddings provided. Use t5_embeddings_path, configure t5_embeddings_dir for auto lookup, "
        "or start the server with --default_t5_embeddings_path."
    )


def run_model_inference(request: InferenceRequest) -> InferenceResponse:
    if SERVER_STATE.model is None or SERVER_STATE.config_dict is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    cfg = SERVER_STATE.config_dict
    common = cfg["common"]
    num_inference_steps = request.num_inference_steps or SERVER_STATE.default_steps
    effective_instruction = resolve_instruction(request)

    start_time = datetime.now()

    # Prepare image
    image_hw = (common["video_height"], common["video_width"])
    input_image = get_input_image(request, image_hw)
    resized_pil = resize_image_with_padding(input_image, image_hw)
    first_frame = image_to_tensor(input_image, image_hw).to(SERVER_STATE.device)

    # Prepare state
    state_dim = int(common["state_dim"])
    state = get_state_tensor(resolve_state_values(request), state_dim, SERVER_STATE.device)

    # Prepare VLM inputs
    vlm_inputs = build_vlm_inputs(SERVER_STATE.processor, effective_instruction, resized_pil, SERVER_STATE.device)
    vlm_inputs = [vlm_inputs]  # Wrap in list

    # Prepare T5 embeddings
    language_embeddings = resolve_language_embeddings(request, effective_instruction)

    # Run inference
    with SERVER_STATE.lock:
        with torch.inference_mode():
            predicted_frames, predicted_actions = SERVER_STATE.model.inference_step(
                first_frame=first_frame,
                state=state,
                num_inference_steps=num_inference_steps,
                language_embeddings=language_embeddings,
                vlm_inputs=vlm_inputs,
            )

    end_time = datetime.now()
    processing_time_ms = (end_time - start_time).total_seconds() * 1000.0

    # Process outputs
    predicted_actions_cpu = predicted_actions.detach().cpu().float()
    if predicted_actions_cpu.dim() == 3:
        predicted_actions_cpu = predicted_actions_cpu.squeeze(0)
    if predicted_actions_cpu.dim() == 1:
        predicted_actions_cpu = predicted_actions_cpu.unsqueeze(0)

    # Denormalize if required (Dobot uses [0,1] normalized actions during training)
    # if SERVER_STATE.action_denorm_required:
    #     action_min = torch.from_numpy(SERVER_STATE.action_min).unsqueeze(0)
    #     action_max = torch.from_numpy(SERVER_STATE.action_max).unsqueeze(0)
    #     predicted_actions_cpu = predicted_actions_cpu * (action_max - action_min) + action_min

    # Create frame grid
    frame_grid_b64 = None
    if request.return_frame_grid:
        first_frame_np = first_frame.squeeze(0).detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
        first_frame_np = (first_frame_np * 255).astype(np.uint8)

        frame_images = [first_frame_np]
        frames_tensor = predicted_frames.squeeze(0).detach().cpu().float().clamp(0, 1)
        for frame in frames_tensor:
            frame_np = frame.permute(1, 2, 0).numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
            frame_images.append(frame_np)

        grid = np.concatenate(frame_images, axis=1)
        grid_image = Image.fromarray(grid)
        buffer = io.BytesIO()
        grid_image.save(buffer, format="PNG")
        frame_grid_b64 = base64.b64encode(buffer.getvalue()).decode()

        if request.save_output_path:
            output_path = Path(request.save_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            grid_image.save(output_path)

    return InferenceResponse(
        predicted_actions=predicted_actions_cpu.tolist(),
        instruction=effective_instruction,
        action_head_type="motus_wan_vlm_direct_mask",
        action_shape=list(predicted_actions_cpu.shape),
        predicted_frames_shape=list(predicted_frames.shape),
        frame_grid_image=frame_grid_b64,
        processing_time_ms=processing_time_ms,
        model_info={
            "device": str(SERVER_STATE.device),
            "num_inference_steps": num_inference_steps,
            "state_dim": state_dim,
            "action_dim": int(common["action_dim"]),
            "video_size": [int(common["video_height"]), int(common["video_width"])],
            "num_video_frames": int(common["num_video_frames"]),
            "default_instruction": SERVER_STATE.default_instruction,
            "default_t5_embeddings_path": SERVER_STATE.default_t5_embeddings_path,
        },
        timestamp=end_time.isoformat(),
    )


def run_mock_inference() -> InferenceResponse:
    start_time = datetime.now()

    if SERVER_STATE.config_dict is not None:
        common = SERVER_STATE.config_dict["common"]
        state_dim = int(common["state_dim"])
        action_dim = int(common["action_dim"])
        num_video_frames = int(common["num_video_frames"])
        video_action_freq_ratio = int(common["video_action_freq_ratio"])
        video_height = int(common["video_height"])
        video_width = int(common["video_width"])
    else:
        state_dim = 14
        action_dim = 14
        num_video_frames = 16
        video_action_freq_ratio = 1
        video_height = 320
        video_width = 384

    action_steps = max(1, num_video_frames * video_action_freq_ratio)
    predicted_actions = torch.randn(action_steps, action_dim, dtype=torch.float32).tolist()

    end_time = datetime.now()
    processing_time_ms = (end_time - start_time).total_seconds() * 1000.0

    return InferenceResponse(
        predicted_actions=predicted_actions,
        instruction=SERVER_STATE.default_instruction or "mock instruction",
        action_head_type="motus_wan_vlm_direct_mask_mock",
        processing_time_ms=processing_time_ms,
        model_info={
            "device": str(SERVER_STATE.device),
            "mock": True,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "video_size": [video_height, video_width],
            "num_video_frames": num_video_frames,
            "default_instruction": SERVER_STATE.default_instruction,
        },
        timestamp=end_time.isoformat(),
        action_shape=[action_steps, action_dim],
        predicted_frames_shape=[1, num_video_frames, 3, video_height, video_width],
        frame_grid_image=None,
    )


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "MotusWanVlmDirectMask Inference API Server",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "inference": "/inference",
            "mock_inference": "/inference/mock",
            "model_info": "/model_info",
            "websocket": "/ws",
        },
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_loaded=SERVER_STATE.model is not None,
        device=str(SERVER_STATE.device),
        timestamp=datetime.now().isoformat(),
    )


@app.get("/model_info")
def model_info() -> Dict[str, Any]:
    if SERVER_STATE.model is None or SERVER_STATE.config_dict is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    common = SERVER_STATE.config_dict["common"]
    return {
        "model_loaded": True,
        "device": str(SERVER_STATE.device),
        "config": {
            "action_head": "motus_wan_vlm_direct_mask",
            "state_dim": int(common["state_dim"]),
            "action_dim": int(common["action_dim"]),
            "video_height": int(common["video_height"]),
            "video_width": int(common["video_width"]),
            "num_video_frames": int(common["num_video_frames"]),
            "video_action_freq_ratio": int(common["video_action_freq_ratio"]),
            "checkpoint_path": SERVER_STATE.checkpoint_path,
            "wan_path": SERVER_STATE.wan_path,
            "vlm_path": SERVER_STATE.vlm_path,
            "default_instruction": SERVER_STATE.default_instruction,
            "default_t5_embeddings_path": SERVER_STATE.default_t5_embeddings_path,
            "t5_embeddings_dir": SERVER_STATE.t5_embeddings_dir,
        },
    }


@app.post("/inference", response_model=InferenceResponse)
def inference(request: InferenceRequest) -> InferenceResponse:
    try:
        return run_model_inference(request)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/inference/mock", response_model=InferenceResponse)
def inference_mock() -> InferenceResponse:
    try:
        return run_mock_inference()
    except Exception as exc:
        log.exception("Mock inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _handle_ws_message(data: Dict[str, Any]) -> Dict[str, Any]:
    msg_type = data.get("type", "inference")
    if msg_type == "health":
        health = HealthResponse(
            status="healthy",
            model_loaded=SERVER_STATE.model is not None,
            device=str(SERVER_STATE.device),
            timestamp=datetime.now().isoformat(),
        )
        return {"type": "health", **health.model_dump()}
    if msg_type == "mock":
        result = await asyncio.to_thread(run_mock_inference)
        return {"type": "mock", **result.model_dump()}
    if msg_type == "inference":
        payload = {k: v for k, v in data.items() if k != "type"}
        request = InferenceRequest(**payload)
        try:
            result = await asyncio.to_thread(run_model_inference, request)
            return {"type": "inference", **result.model_dump()}
        except HTTPException as exc:
            return {"type": "error", "detail": exc.detail}
        except Exception as exc:
            log.exception("WebSocket inference failed")
            return {"type": "error", "detail": str(exc)}
    return {"type": "error", "detail": f"Unknown message type: {msg_type}"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            response = await _handle_ws_message(data)
            await websocket.send_json(response)
    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    except Exception as exc:
        log.exception("WebSocket handler failed")
        try:
            await websocket.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MotusWanVlmDirectMask Inference API Server")
    parser.add_argument("--model_config", required=True, help="Path to YAML config")
    parser.add_argument("--ckpt_dir", required=True, help="Path to checkpoint directory")
    parser.add_argument("--wan_path", required=True, help="WAN model path")
    parser.add_argument("--vlm_path", default=None, help="VLM path override")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0")
    parser.add_argument("--t5_embeddings_dir", default=None, help="Directory for auto T5 lookup")
    parser.add_argument("--default_instruction", default=None, help="Default instruction used when a request omits it")
    parser.add_argument("--default_instruction_file", default=None,
                        help="Path to a text file containing the default instruction")
    parser.add_argument("--default_t5_embeddings_path", default=None,
                        help="Precompiled default T5 embedding .pt file used when a request omits it")
    parser.add_argument("--dataset_name", default=None, choices=list(DATASET_ACTION_STATS.keys()),
                        help="Dataset name for auto-loading action min/max (e.g., dobot_pour_water, dobot_cook_vegetable)")
    parser.add_argument("--action_min", type=float, nargs='+', default=None,
                        help="Action min values for denormalization (overrides dataset_name)")
    parser.add_argument("--action_max", type=float, nargs='+', default=None,
                        help="Action max values for denormalization (overrides dataset_name)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=6789, type=int)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_argparser().parse_args()

    # Convert action_min/max to numpy arrays
    action_min = np.array(args.action_min, dtype=np.float32) if args.action_min else None
    action_max = np.array(args.action_max, dtype=np.float32) if args.action_max else None

    # Auto-load from dataset_name if not explicitly provided
    if args.dataset_name and action_min is None and action_max is None:
        stats = DATASET_ACTION_STATS[args.dataset_name]
        action_min = np.array(stats["min"], dtype=np.float32)
        action_max = np.array(stats["max"], dtype=np.float32)
        log.info(f"Loaded action stats for dataset '{args.dataset_name}' from built-in stats")

    load_server_components(
        model_config_path=args.model_config,
        ckpt_dir=args.ckpt_dir,
        wan_path=args.wan_path,
        vlm_path=args.vlm_path,
        device=args.device,
        t5_embeddings_dir=args.t5_embeddings_dir,
        default_instruction=args.default_instruction,
        default_instruction_file=args.default_instruction_file,
        default_t5_embeddings_path=args.default_t5_embeddings_path,
        action_min=action_min,
        action_max=action_max,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    import sys
    main()
