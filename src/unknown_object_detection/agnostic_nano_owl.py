'''
Author: 
Description: 

Make sure to install these into your environment/wherever your dependencies are to get OWL started: 

python -m pip install "git+https://github.com/NVIDIA-AI-IOT/nanoowl.git"
python -m pip install torch torchvision transformers pillow numpy
'''

#imports
import cv2
from PIL import Image
from nanoowl.owl_predictor import OwlPredictor

#owl predictor model
predictor = OwlPredictor(
    "google/owlvit-base-patch32",
    device="cpu"
)
