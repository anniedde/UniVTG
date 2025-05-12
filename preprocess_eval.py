import os
import json
import ffmpeg
from tqdm import tqdm

# === CONFIGURATION ===
input_scanqa_json = '/playpen-nas-ssd4/awang/GPT4Scene/evaluate/annotation/scanqa_val.json'  # Path to your input JSON file
videos_dir = '/playpen-nas-ssd4/awang/scannet_scenes/videos'  # Path to your videos
output_jsonl = '/playpen-nas-ssd4/awang/UniVTG/eval.jsonl'  # Output JSONL file

# === FUNCTION TO GET VIDEO DURATION ===
def get_video_duration(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        return duration
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

# === LOAD INPUT JSON ===
with open(input_scanqa_json, 'r') as f:
    scanqa_data = json.load(f)

# === BUILD JSONL ENTRIES ===
jsonl_entries = []

for entry in tqdm(scanqa_data, desc="Processing entries"):
    scene_id = entry.get('scene_id')
    qid = entry.get('qid')
    prompt = entry.get('prompt')

    # Clean up the prompt (optional)
    cleaned_prompt = prompt.replace(' Answer the question using a single word or phrase.', '').strip()

    # Build the video path
    video_filename = f"{scene_id}.mp4"
    video_path = os.path.join(videos_dir, video_filename)

    # Get video duration
    duration = get_video_duration(video_path)

    if duration is None:
        print(f"Skipping entry with scene_id: {scene_id} (missing duration)")
        continue  # skip if we can't get the duration

    # Build JSONL record
    jsonl_entry = {
        "qid": qid,
        "vid": scene_id,
        "query": cleaned_prompt,
        "duration": duration
    }

    jsonl_entries.append(jsonl_entry)

# === SAVE JSONL FILE ===
with open(output_jsonl, 'w') as f:
    for item in jsonl_entries:
        f.write(json.dumps(item) + '\n')

print(f"\n✅ Finished! JSONL file saved at: {output_jsonl}")