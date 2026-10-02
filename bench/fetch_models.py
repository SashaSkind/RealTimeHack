"""Download the local Cosmos Reason 2 GGUF weights (third-party Q8_0 conversions under the NVIDIA
Open Model License) into models/. YOLO weights, the MediaPipe hand model and Depth Anything download
on first use; the MediaPipe model URL is below.

  python bench/fetch_models.py
  curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
"""
from huggingface_hub import hf_hub_download
import time
jobs=[("apolo13x/Cosmos-Reason2-2B-GGUF","Cosmos-Reason2-2B-Q8_0.gguf"),
      ("apolo13x/Cosmos-Reason2-2B-GGUF","mmproj-Cosmos-Reason2-2B-F16.gguf"),
      ("mradermacher/Cosmos-Reason2-8B-GGUF","Cosmos-Reason2-8B.mmproj-f16.gguf"),
      ("mradermacher/Cosmos-Reason2-8B-GGUF","Cosmos-Reason2-8B.Q8_0.gguf")]
for rid,f in jobs:
    t=time.time(); p=hf_hub_download(rid,f,local_dir="models/"+rid.split("/")[1]); print(p, round(time.time()-t),"s",flush=True)
