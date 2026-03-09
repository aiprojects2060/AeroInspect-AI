"""
BLIP Image Captioning and Summarization Module
Tasks 8-10 from the final project
Pure PyTorch/Transformers — no TensorFlow required.
"""
import os, sys

# ─── Add local pip packages to path (transformers, torch, etc.) ───────────────
_pip_pkg_dir = r"C:\Users\Orion\pip_packages"
if os.path.isdir(_pip_pkg_dir) and _pip_pkg_dir not in sys.path:
    sys.path.insert(0, _pip_pkg_dir)

import warnings
warnings.filterwarnings('ignore')

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


# ── Global model cache ─────────────────────────────────────────────────────────
_processor = None
_blip_model = None


def load_blip_model():
    """2.1 – Load the BLIP processor and model from Hugging Face."""
    global _processor, _blip_model
    if _processor is None or _blip_model is None:
        _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return _processor, _blip_model


# ── Task 8: BlipCaptionSummaryLayer (pure Python class, no TF) ────────────────
class BlipCaptionSummaryLayer:
    """
    Custom layer wrapping the BLIP model — mirrors the Keras layer interface
    described in Task 8, implemented in pure PyTorch so TensorFlow is not needed.
    """

    def __init__(self, processor, model):
        self.processor = processor
        self.model = model

    def __call__(self, image_path: str, task: str) -> str:
        return self.process_image(image_path, task)

    def process_image(self, image_path: str, task: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")

            if task == "caption":
                prompt = "This is a picture of"
            else:
                prompt = "This is a detailed photo showing"

            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=50)
            result = self.processor.decode(output[0], skip_special_tokens=True)
            return result
        except Exception as e:
            print(f"Error: {e}")
            return "Error processing image"


# ── Task 8: Helper function ────────────────────────────────────────────────────
def generate_text(image_path: str, task: str) -> str:
    """
    Task 8 – Helper function that instantiates BlipCaptionSummaryLayer
    and returns generated caption or summary text.
    """
    processor, model = load_blip_model()
    blip_layer = BlipCaptionSummaryLayer(processor, model)
    return blip_layer(image_path, task)


# ── Tasks 9 & 10 ──────────────────────────────────────────────────────────────
def generate_caption_and_summary(image_path_str: str):
    """
    Tasks 9 & 10 – Generate caption and summary for a given image path.
    Returns (caption, summary) as Python strings.
    """
    caption = generate_text(image_path_str, "caption")
    summary = generate_text(image_path_str, "summary")
    return caption, summary


def generate_from_pil_image(pil_image, task: str = "caption") -> str:
    """Generate caption/summary directly from a PIL Image (for Streamlit uploads)."""
    processor, model = load_blip_model()

    if task == "caption":
        prompt = "This is a picture of"
    else:
        prompt = "This is a detailed photo showing"

    inputs = processor(images=pil_image.convert("RGB"), text=prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    result = processor.decode(output[0], skip_special_tokens=True)
    return result
