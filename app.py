from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import torch
import os
import time
import logging
from typing import Optional

from acestep.pipeline_ace_step import ACEStepPipeline

from torch import nn

if not hasattr(nn, "RMSNorm"):
    class RMSNorm(nn.Module):
        """
        Drop‑in replacement for torch.nn.RMSNorm (PyTorch ≥ 2.0).
        Matches HF implementation except `elementwise_affine=False` by default
        (ACE‑Step sets it that way).
        """
        def __init__(self, hidden_size, eps=1e-6, elementwise_affine=False):
            super().__init__()
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(hidden_size))
            else:
                self.register_parameter("weight", None)
            self.hidden_size = hidden_size

        def forward(self, x):
            # variance across the last dimension
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            if self.weight is not None:
                x = x * self.weight
            return x

    nn.RMSNorm = RMSNorm  # <‑‑ makes ACE‑Step happy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lyrics-to-music-api")

app = FastAPI(title="Lyrics to Music API", description="Generate music from lyrics using ACEStep model")

# Define request model with all parameters
class LyricsToMusicRequest(BaseModel):
    lyrics: str
    prompt: str
    
    # Optional parameters with default values
    audio_duration: float = Field(default=60.0, description="Duration of the output audio in seconds")
    infer_step: int = Field(default=60, description="Number of inference steps")
    guidance_scale: float = Field(default=15.0, description="Guidance scale for the model")
    scheduler_type: str = Field(default="euler", description="Scheduler type (euler or heun)")
    cfg_type: str = Field(default="apg", description="CFG type (apg, cfg, or cfg_star)")
    omega_scale: int = Field(default=10, description="Omega scale parameter")
    manual_seeds: Optional[str] = Field(default=None, description="Comma-separated seeds or single seed")
    guidance_interval: float = Field(default=0.5, description="Guidance interval parameter")
    guidance_interval_decay: float = Field(default=0.0, description="Guidance interval decay parameter")
    min_guidance_scale: float = Field(default=3.0, description="Minimum guidance scale")
    use_erg_tag: bool = Field(default=True, description="Whether to use ERG tag")
    use_erg_lyric: bool = Field(default=True, description="Whether to use ERG lyric")
    use_erg_diffusion: bool = Field(default=True, description="Whether to use ERG diffusion")
    oss_steps: Optional[str] = Field(default=None, description="OSS steps as comma-separated integers")
    guidance_scale_text: float = Field(default=0.0, description="Guidance scale for text")
    guidance_scale_lyric: float = Field(default=0.0, description="Guidance scale for lyrics")

# Global variables
model = None
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Initialize the model
def get_model():
    global model
    
    if model is None:
        logger.info("Loading ACEStepPipeline model...")
        model = ACEStepPipeline(
            checkpoint_dir="ACE-Step-v1-3.5B",  # Use default path
            dtype="bfloat16", 
            torch_compile=False,
        )
        logger.info("Model loaded successfully")
    
    return model

@app.on_event("startup")
async def startup_event():
    # Pre-load the model on startup
    get_model()

# Function to clean up after generation
def cleanup(file_path=None):
    try:
        # Empty CUDA cache
        torch.cuda.empty_cache()
        logger.info("CUDA cache emptied")
        
        # Clean up generated files if path is provided
        if file_path and os.path.exists(file_path):
            # Remove the audio file
            os.remove(file_path)
            
            # Remove the associated JSON file if it exists
            json_path = file_path.replace(".wav", "_input_params.json")
            if os.path.exists(json_path):
                os.remove(json_path)
                
            logger.info(f"Cleaned up generated files: {file_path}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

@app.post("/generate-music-ace")
async def generate_music(request: LyricsToMusicRequest, background_tasks: BackgroundTasks):
    """
    Generate music from lyrics and a prompt.
    
    Args:
        request: The request containing lyrics, prompt, and optional parameters
        
    Returns:
        The generated audio file as a direct download
    """
    # Get the model
    model = get_model()
    
    try:
        # Create a unique output file path
        timestamp = time.strftime("%Y%m%d%H%M%S")
        output_file = f"{output_dir}/music_{timestamp}.wav"
        
        logger.info(f"Generating music with lyrics: {request.lyrics[:50]}... and prompt: {request.prompt[:50]}...")
        
        # Call the model with specified parameters
        result = model(
            audio_duration=request.audio_duration,
            prompt=request.prompt,
            lyrics=request.lyrics,
            infer_step=request.infer_step,
            guidance_scale=request.guidance_scale,
            scheduler_type=request.scheduler_type,
            cfg_type=request.cfg_type,
            omega_scale=request.omega_scale,
            manual_seeds=request.manual_seeds,
            guidance_interval=request.guidance_interval,
            guidance_interval_decay=request.guidance_interval_decay,
            min_guidance_scale=request.min_guidance_scale,
            use_erg_tag=request.use_erg_tag,
            use_erg_lyric=request.use_erg_lyric,
            use_erg_diffusion=request.use_erg_diffusion,
            oss_steps=request.oss_steps,
            guidance_scale_text=request.guidance_scale_text,
            guidance_scale_lyric=request.guidance_scale_lyric,
            save_path=output_file,
        )
        
        # The first element in the result should be the audio file path
        audio_file_path = result[0]
        logger.info(f"Music generation complete, audio saved to: {audio_file_path}")
        
        # Return the audio file directly and schedule cleanup to run after sending the response
        response = FileResponse(
            path=audio_file_path,
            filename=os.path.basename(audio_file_path),
            media_type="audio/wav"
        )
        
        # Schedule cleanup to run after sending the response (with the file path for deletion)
        background_tasks.add_task(cleanup, audio_file_path)
        
        return response
        
    except Exception as e:
        logger.error(f"Music generation failed: {str(e)}")
        # Clean up on error
        cleanup()
        raise HTTPException(status_code=500, detail=f"Music generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8060)
