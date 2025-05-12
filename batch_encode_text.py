import os
import json
import torch
import argparse
import logging
import numpy as np
from run_on_video import clip, txt2clip
from tqdm import tqdm

# =========================
# Setup Logger and Args
# =========================
parser = argparse.ArgumentParser(description='Extract text features from prompts and save them.')
parser.add_argument('--prompts_loc', type=str, required=True,
                    help='Path to JSONL file containing prompts.')
parser.add_argument('--output_dir', type=str, default='./text_features',
                    help='Directory where text features will be saved.')
parser.add_argument("--gpu_id", type=int, default=2)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

# =========================
# Load CLIP Model
# =========================
model_version = "ViT-B/32"
logger.info(f"Loading CLIP model: {model_version}...")
clip_model, _ = clip.load(model_version, device=args.gpu_id, jit=False)

# =========================
# Load Prompts
# =========================
prompts_loc = args.prompts_loc
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

logger.info(f"Loading prompts from {prompts_loc}")
prompts_data = []
with open(prompts_loc, 'r') as f:
    for line in f:
        prompts_data.append(json.loads(line.strip()))

logger.info(f"Total prompts found: {len(prompts_data)}")

# =========================
# Extract Text Features
# =========================
text_features_dict = {}

for entry in tqdm(prompts_data):
    qid = entry.get('qid')
    query = entry.get('query')

    if qid is None or query is None:
        logger.warning(f"Skipping entry with missing qid or query: {entry}")
        continue

    # Extract text feature and save
    txt2clip(clip_model, query, output_dir, save_name=f'{qid}.npz')

    # Load saved text feature from the npz
    txt_feat_path = os.path.join(output_dir, f"{qid}.npz")