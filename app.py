# filename: app.py
import os
import uuid
import tempfile
import shutil
import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.data_sampler import DataSampler


# ----------------------------------------------------------------------
# 1. Model & helper utilities
# ----------------------------------------------------------------------

def sample_data(json_data):
    return (
        json_data["audio_duration"],
        json_data["prompt"],
        json_data["lyrics"],
        json_data["infer_step"],
        json_data["guidance_scale"],
        json_data["scheduler_type"],
        json_data["cfg_type"],
        json_data["omega_scale"],
        ", ".join(map(str, json_data["actual_seeds"])),
        json_data["guidance_interval"],
        json_data["guidance_interval_decay"],
        json_data["min_guidance_scale"],
        json_data["use_erg_tag"],
        json_data["use_erg_lyric"],
        json_data["use_erg_diffusion"],
        ", ".join(map(str, json_data["oss_steps"])),
        json_data["guidance_scale_text"] if "guidance_scale_text" in json_data else 0.0,
        (
            json_data["guidance_scale_lyric"]
            if "guidance_scale_lyric" in json_data
            else 0.0
        ),
    )

def _sample_default_args() -> tuple:
    """
    Use the library's own DataSampler to obtain a dict of all default
    arguments; convert it with sample_data() exactly as the original demo did.
    """
    return sample_data(DataSampler().sample())


# Load model **once** at startup (change env var if you keep checkpoints elsewhere)
CHECKPOINT_DIR = "ACE-Step-v1-3.5B" #os.getenv("CHECKPOINT_PATH", "")
MODEL = ACEStepPipeline(
    checkpoint_dir=CHECKPOINT_DIR,
    dtype="bfloat16",          # library default
    torch_compile=False        # safer for first deployment; enable later if desired
)


# ----------------------------------------------------------------------
# 2. FastAPI definitions
# ----------------------------------------------------------------------
app = FastAPI(
    title="Lyrics‑to‑Music API",
    description="Generate a complete music track from lyrics + text prompt.",
    version="1.0.0",
)


class Lyrics2MusicRequest(BaseModel):
    lyrics: str
    prompt: str


@app.post("/generate_music_ace", summary="Generate music from lyrics", response_class=FileResponse)
async def generate_music(body: Lyrics2MusicRequest):
    """
    Accepts JSON:
        {"lyrics": "...", "prompt": "…"}
    Returns: audio/wav file containing the generated music.
    """
    # ------------------------------------------------------------------
    # prepare arguments (defaults + user text)
    # ------------------------------------------------------------------
    (
        audio_duration,
        _prompt,                 # will be overwritten
        _lyrics,                 # will be overwritten
        infer_step,
        guidance_scale,
        scheduler_type,
        cfg_type,
        omega_scale,
        manual_seeds,
        guidance_interval,
        guidance_interval_decay,
        min_guidance_scale,
        use_erg_tag,
        use_erg_lyric,
        use_erg_diffusion,
        oss_steps,
        guidance_scale_text,
        guidance_scale_lyric,
    ) = _sample_default_args()

    prompt = body.prompt
    lyrics = body.lyrics

    # ------------------------------------------------------------------
    # create a temp .wav for output
    # ------------------------------------------------------------------
    tmp_dir = tempfile.mkdtemp()
    outfile = os.path.join(tmp_dir, f"{uuid.uuid4()}.wav")

    try:
        # ----------------------  run generation  ----------------------
        MODEL(
            audio_duration,
            prompt,
            lyrics,
            infer_step,
            guidance_scale,
            scheduler_type,
            cfg_type,
            omega_scale,
            manual_seeds,
            guidance_interval,
            guidance_interval_decay,
            min_guidance_scale,
            use_erg_tag,
            use_erg_lyric,
            use_erg_diffusion,
            oss_steps,
            guidance_scale_text,
            guidance_scale_lyric,
            save_path=outfile,
        )
        torch.cuda.empty_cache()

        # ----------------------  return file  -------------------------
        return FileResponse(
            outfile,
            media_type="audio/wav",
            filename="lyrics2music.wav",
            background=lambda *_: shutil.rmtree(tmp_dir, ignore_errors=True),
        )

    except Exception as e:
        torch.cuda.empty_cache()
        # clean up temp directory on errors as well
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# 3. Uvicorn entry point
# ----------------------------------------------------------------------
# Run with:
#   CHECKPOINT_PATH=/path/to/checkpoint \
#   uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
#
# Increase --workers if you need multiple *processes* (each with its own GPU),
# or use --reload during development for auto‑restart.
