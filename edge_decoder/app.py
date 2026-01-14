import torch
import numpy as np
from fastapi import FastAPI, Body
from fastapi.responses import Response
from PIL import Image
import io
import os
import logging
from utils.models import VAEWrapper
import torchvision.transforms as transforms


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EdgeDecoder")

app = FastAPI()

# Global model variable
decoder = None

def load_model():
    global decoder
    logger.info("Initializing Edge Decoder (VAE)...")
    decoder = VAEWrapper(device='cpu')

    
    # Removed legacy weight loading. VAE loads itself.



@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/decode")
async def decode_vector(payload: dict = Body(...)):
    try:
        vector_list = payload.get("vector")
        if not vector_list:
            return {"error": "No vector provided"}
            
        vector_np = np.array(vector_list, dtype=np.float32)
        # Reshape if flat? VAE expects [B, 4, H, W]
        # If EdgeEncoder sent nested list, we are good.
        if vector_np.ndim == 1:
            # Fallback reshape if flattened (4 * 32 * 32 = 4096)
            vector_np = vector_np.reshape(4, 32, 32)
            
        vector_t = torch.from_numpy(vector_np).unsqueeze(0) # Add batch dim
        
        with torch.no_grad():
            reconstructed_img_t = decoder.decode(vector_t)
            # Output is [1, 3, 256, 256] in [0, 1]
            
            # Convert to PIL Image
            img_t = reconstructed_img_t.squeeze(0)

            img_pil = transforms.ToPILImage()(img_t)
            
            # Save to bytes
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='JPEG')
            return Response(content=img_byte_arr.getvalue(), media_type="image/jpeg")
            
    except Exception as e:
        logger.error(f"Decoding error: {e}")
        return {"error": str(e)}
