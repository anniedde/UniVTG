import os
import time
import json
import torch
import argparse
import logging
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from collections import defaultdict
from run_on_video import clip, vid2clip, txt2clip
from main.config import TestOptions, setup_model
from utils.basic_utils import l2_normalize_np_array
from tqdm import tqdm

# =========================
# Setup Logger and Args
# =========================
parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_loc', type=str, default='./selected_frames.json')
parser.add_argument('--resume', type=str, default='./results/omni/model_best.ckpt')
parser.add_argument("--gpu_id", type=int, default=2)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

# =========================
# Load Models
# =========================
def load_model():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse(args)
    cudnn.benchmark = True
    cudnn.deterministic = False

    if opt.lr_warmup > 0:
        total_steps = opt.n_epoch
        warmup_steps = opt.lr_warmup if opt.lr_warmup > 1 else int(opt.lr_warmup * total_steps)
        opt.lr_warmup = [warmup_steps, total_steps]

    model, criterion, _, _ = setup_model(opt)
    return model

model_version = "ViT-B/32"
clip_len = 2
save_dir = './tmp'

logger.info("Loading models...")
vtg_model = load_model()
clip_model, _ = clip.load(model_version, device=args.gpu_id, jit=False)

# =========================
# Dataset 
# =========================
videos_loc = "/playpen-nas-ssd4/awang/scannet_scenes/videos"
prompts_loc = "/playpen-nas-ssd4/awang/GPT4Scene/evaluate/annotation/scanqa_val.json"

# Initialize dataset list
dataset = []

# Load the JSON file
with open(prompts_loc, 'r') as f:
    prompts_data = json.load(f)

# Iterate through each entry and build the dataset
for entry in prompts_data:
    scene_id = entry.get('scene_id')
    prompt = entry.get('prompt').replace(' Answer the question using a single word or phrase.', '')

    if not scene_id or not prompt:
        continue  # skip if missing data

    video_filename = f"{scene_id}.mp4"
    video_path = os.path.join(videos_loc, video_filename)

    # Append to dataset
    dataset.append({
        "video_path": video_path,
        "query": prompt
    })

# OPTIONAL: print or check the dataset
#for item in dataset[:5]:  # Show first 5 entries
#    print(item)

# =========================
# Merge Intervals Helper
# =========================
def merge_intervals(intervals):
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)
    return merged

# =========================
# Batch Processing Loop
# =========================

def convert_to_hms(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def load_data(save_dir):
    vid = np.load(os.path.join(save_dir, 'vid.npz'))['features'].astype(np.float32)
    txt = np.load(os.path.join(save_dir, 'txt.npz'))['features'].astype(np.float32)

    vid = torch.from_numpy(l2_normalize_np_array(vid))
    txt = torch.from_numpy(l2_normalize_np_array(txt))
    clip_len = 2
    ctx_l = vid.shape[0]

    timestamp =  ( (torch.arange(0, ctx_l) + clip_len / 2) / ctx_l).unsqueeze(1).repeat(1, 2)

    if True:
        tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
        tef_ed = tef_st + 1.0 / ctx_l
        tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
        vid = torch.cat([vid, tef], dim=1)  # (Lv, Dv+2)

    src_vid = vid.unsqueeze(0).cuda()
    src_txt = txt.unsqueeze(0).cuda()
    src_vid_mask = torch.ones(src_vid.shape[0], src_vid.shape[1]).cuda()
    src_txt_mask = torch.ones(src_txt.shape[0], src_txt.shape[1]).cuda()

    return src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l

def forward(model, save_dir, query):
    src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l = load_data(save_dir)
    src_vid = src_vid.cuda(args.gpu_id)
    src_txt = src_txt.cuda(args.gpu_id)
    src_vid_mask = src_vid_mask.cuda(args.gpu_id)
    src_txt_mask = src_txt_mask.cuda(args.gpu_id)

    model.eval()
    with torch.no_grad():
        output = model(src_vid=src_vid, src_txt=src_txt, src_vid_mask=src_vid_mask, src_txt_mask=src_txt_mask)

    # Add timestamp + ctx_l + clip_len because you need them later
    return output, timestamp, ctx_l

results = []
coverage_percentages = []
total_covered_times = []
top_confidences_by_rank = [[] for _ in range(5)]  # Top-1 to Top-5 lists

for item in tqdm(dataset):
    video_path = item['video_path']
    query = item['query']
    print(f"\nProcessing video: {video_path} | Query: {query}")

    # Step 1: Extract video features
    vid2clip(clip_model, video_path, save_dir)

    # Step 2: Extract text features
    txt2clip(clip_model, query, save_dir)

    # Step 3: Run model forward pass
    output, timestamp, ctx_l = forward(vtg_model, save_dir, query)

    # Extract and process top-5 intervals
    src_vid, _, _, _, _, ctx_l = load_data(save_dir)
    video_duration = ctx_l * clip_len  # in seconds

    pred_logits = output['pred_logits'][0].cpu()
    pred_spans = output['pred_spans'][0].cpu()

    timestamp = ((torch.arange(0, ctx_l) + clip_len / 2) / ctx_l).unsqueeze(1).repeat(1, 2)
    pred_windows = (pred_spans + timestamp) * ctx_l * clip_len
    pred_confidence = pred_logits

    top5_values, top5_indices = torch.topk(pred_confidence.flatten(), k=5)
    top5_windows = pred_windows[top5_indices].tolist()
    top5_confidences = top5_values.tolist()

    # Merge intervals and compute time coverage
    merged_intervals = merge_intervals([[int(window[0]), int(window[1])] for window in top5_windows])
    total_covered_time = sum([end - start for start, end in merged_intervals])
    coverage_percentage = (total_covered_time / video_duration) * 100 if video_duration > 0 else 0

    # Store per-video results
    result = {
        "video_path": video_path,
        "query": query,
        "video_duration": video_duration,
        "top5_intervals": top5_windows,
        "top5_confidences": top5_confidences,
        "merged_intervals": merged_intervals,
        "total_covered_time": total_covered_time,
        "coverage_percentage": coverage_percentage
    }

    results.append(result)

    # Aggregate lists
    coverage_percentages.append(coverage_percentage)
    total_covered_times.append(total_covered_time)
    for rank, confidence in enumerate(top5_confidences):
        top_confidences_by_rank[rank].append(confidence)

    # Debug print
    print(f"Results for {video_path}:")
    print(f"  Coverage %: {coverage_percentage:.2f}%")
    print(f"  Total Covered Time: {total_covered_time}s")
    print(f"  Top-5 Confidences: {top5_confidences}")

# =========================
# Aggregated Stats
# =========================
coverage_percentages = np.array(coverage_percentages)
total_covered_times = np.array(total_covered_times)

# Overall aggregates
aggregate_stats = {
    "coverage_percentage": {
        "mean": float(np.mean(coverage_percentages)),
        "min": float(np.min(coverage_percentages)),
        "max": float(np.max(coverage_percentages)),
        "std": float(np.std(coverage_percentages)),
    },
    "total_covered_time": {
        "mean": float(np.mean(total_covered_times)),
        "min": float(np.min(total_covered_times)),
        "max": float(np.max(total_covered_times)),
        "std": float(np.std(total_covered_times)),
    }
}

# Per-rank aggregates (Top-1, Top-2, ..., Top-5)
aggregate_ranked_confidences = {}
for rank in range(5):
    confidences = np.array(top_confidences_by_rank[rank])
    aggregate_ranked_confidences[f"top{rank+1}_confidence"] = {
        "mean": float(np.mean(confidences)),
        "min": float(np.min(confidences)),
        "max": float(np.max(confidences)),
        "std": float(np.std(confidences)),
    }

aggregate_stats["ranked_top5_confidences"] = aggregate_ranked_confidences

# =========================
# Save Results
# =========================
with open("batch_inference_results.json", "w") as f:
    json.dump(results, f, indent=4)

with open("aggregate_stats.json", "w") as f:
    json.dump(aggregate_stats, f, indent=4)

print("\n=== Aggregate Stats ===")
print(json.dumps(aggregate_stats, indent=4))