"""Repository layout and external tools. Every entry can be overridden with an environment variable.

    TRELLIS_PYTHON          Python of the TRELLIS / Easy3E environment (default: the current interpreter)
    TRELLIS_CHECKPOINT      local TRELLIS-image-large directory (pipeline.json + ckpts/), see README
    TRELLIS_ENC_PRETRAINED  SLAT encoder (default: <TRELLIS_CHECKPOINT>/ckpts/slat_enc_swin8_B_64l8_fp16)
    BLENDER_PATH            Blender 4.0 executable (default: Easy3E/tmp/blender-4.0.0-linux-x64/blender)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MV_FUSION_DIR = REPO_ROOT / "mv_fusion"
EASY3E_ROOT = REPO_ROOT / "Easy3E"
API_DIR = REPO_ROOT / "api"
BLENDER_SCRIPTS_DIR = REPO_ROOT / "blender_scripts"

PYTHON_BIN = Path(os.environ.get("TRELLIS_PYTHON", sys.executable))
CHECKPOINT = Path(os.environ.get("TRELLIS_CHECKPOINT", REPO_ROOT / "checkpoints" / "TRELLIS-image-large"))
ENC_PRETRAINED = Path(os.environ.get("TRELLIS_ENC_PRETRAINED", CHECKPOINT / "ckpts" / "slat_enc_swin8_B_64l8_fp16"))
BLENDER_BIN = Path(os.environ.get("BLENDER_PATH", EASY3E_ROOT / "tmp" / "blender-4.0.0-linux-x64" / "blender"))

RENDER_GLB_MV_SCRIPT = BLENDER_SCRIPTS_DIR / "render_glb_mv.py"
RENDER_GLB_MV_HIGH_DETAIL_SCRIPT = BLENDER_SCRIPTS_DIR / "render_glb_mv_high_detail.py"
