import modal
from pathlib import Path
from fastapi import UploadFile, File


# Modal App & Image setup
app = modal.App("proteus")

finetune_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .run_commands("pip install --upgrade pip")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("wandb", "fastapi")
)
with finetune_image.imports():
    from utils.helper import LinearSVM

wandb_secret = modal.Secret.from_name("wandb-secret")
out_volume = modal.Volume.from_name("out-volume", create_if_missing=True)
OUT_PATH = Path("/out_volume")

@app.cls(
    image=finetune_image,
    gpu=["T4","L4", "A10G"],
    volumes={OUT_PATH: out_volume},
    timeout=300,
    # min_containers=1,
)
class ModelApp:
    def __init__(self):
        self.model = None
        self.svm = None
        self.preprocess = None
        self.device = "cuda" 

    # @modal.exit()
    # def cleanup(self):
    #     self.model = None
    #     self.svm = None
    #     self.preprocess = None
    #     print("🧹 Model cleared on container shutdown")

    @modal.fastapi_endpoint(method="POST")
    async def load_model(self):
        import torch
        import os
        import open_clip
        

        if self.model is None:
            print(" Loading model on user request...")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14-quickgelu", pretrained="dfn5b"
            )
            self.model.visual.proj = None
            self.model.to(self.device)

            self.svm = LinearSVM(in_features=1280).to(self.device)

            checkpoint = torch.load(
                OUT_PATH / "output_adv/full_model" / "joint_model_ViT-H-14-quickgelu_adv20000_epoch9.pth",
                map_location=self.device
            )
            self.model.load_state_dict(checkpoint["clip_model"])
            self.svm.load_state_dict(checkpoint["svm_model"])
            self.model.eval()
            self.svm.eval()

            return {"status": "Model loaded"}
        else:
            return {"status": "Model already loaded"}

    @modal.fastapi_endpoint(method="POST")
    async def infer(self, file: UploadFile = File(...)):
        import torch
        import io
        from PIL import Image

        if self.model is None:
            return {"error": "Model not loaded. Please call /load_model first."}

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            feat = self.model.visual.forward(input_tensor)
            logit = self.svm(feat).squeeze().cpu()
            prob = torch.sigmoid(logit)

        return {"score": float(prob.item())}

    @modal.fastapi_endpoint(method="POST")
    async def reset_model(self):
        import os
        import sys
        import time
        self.model = None
        self.svm = None
        self.preprocess = None

        print("Shutting down container...")
        time.sleep(1) 
        sys.exit(0)

        return {"status": "Model reset"}
