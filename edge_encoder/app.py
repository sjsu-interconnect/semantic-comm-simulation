import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import os
import logging
from utils.models import VAEWrapper
import torchvision.transforms as transforms


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EdgeEncoder")

app = FastAPI()

# Global model variable
encoder = None
preprocess = None

def load_model():
    global encoder, preprocess
    logger.info("Initializing Edge Encoder (VAE)...")
    encoder = VAEWrapper(device='cpu')

    
    # Removed legacy ResNet weight loading. VAE loads itself.

    
    # Resize to 256x256 for VAE
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
    ])


@app.on_event("startup")
async def startup_event():
    load_model()



@app.post("/encode")
async def encode_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)
        
        with torch.no_grad():

            # encoder.encode returns latent [1, 4, 32, 32] (assuming 256x256 in)
            latent = encoder.encode(batch_t)
            vector_np = latent.squeeze().flatten().tolist() # Flatten for JSON transport?
            # Or keep as nested list? Receiver expects numpy array constructable from list.
            # Flattening is safest for JSON logic, but Receiver needs to reshape.
            # Sender sends pure bytes for SEM_EDGE, wait.
            # Sender: _get_edge_feature_vector
            #   response.json()['vector'] -> vector = np.array(data['vector'])
            #   If we flatten here, sender gets flat array.
            #   Sender then creates payload. Receiver gets flat array.
            #   Receiver: _decode_edge_vector -> sends vector to Edge Decoder.
            #   Receiver: _decode_vector -> Reshapes to (4, 32, 32).
            #   Wait, Sender VAE returns (4,32,32).
            #   Edge Encoder JSON must be consistent.
            #   Let's keep it nested list: latent.squeeze().tolist() -> 4x32x32 list.
            vector_np = latent.squeeze().tolist() 

            
        return {"vector": vector_np}
    except Exception as e:
        logger.error(f"Encoding error: {e}")
        return {"error": str(e)}
