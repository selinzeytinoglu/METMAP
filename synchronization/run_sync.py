# run_sync.py — Batch video synchronization using the lights-on protocol.
#
# For each participant, detects the lights-on event in both a reference and a target
# recording and outputs the temporal offset between them. Apply the offset to any
# event timestamp in the reference video to find the corresponding time in the target.
#
# Input CSV columns:
#   participant_id   Unique label for each recording session.
#   reference_video  Path to the reference recording (e.g. room camera).
#   target_video     Path to the target recording (e.g. eye-tracker export).
#
# Output CSV columns:
#   participant_id, reference_lights_on_sec, target_lights_on_sec, offset_sec, notes
#
# To translate an event: target_time = reference_time + offset_sec

import pandas as pd
from sync import find_lights_on

# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------
INPUT_CSV = "participants.csv"
OUTPUT_CSV = "sync_results.csv"

# Mean pixel intensity (0–255) below which a frame is treated as dark.
# Increase if your room lights don't dim fully; decrease if false positives occur.
BRIGHTNESS_THRESHOLD = 50

# Seconds into each video to start scanning (skips camera startup artifacts).
SEARCH_START_SEC = 2.0

# Seconds into each video to stop scanning. Set to None to scan the full video.
SEARCH_END_SEC = None
# ---------------------------------------------------------------------------


def process_row(row: pd.Series) -> dict:
    pid = row["participant_id"]
    ref_video = row["reference_video"]
    tgt_video = row["target_video"]

    result = {
        "participant_id": pid,
        "reference_lights_on_sec": None,
        "target_lights_on_sec": None,
        "offset_sec": None,
        "notes": "",
    }

    # Step 1: find lights-on in the reference video
    try:
        ref_lo_sec, _ = find_lights_on(
            ref_video, SEARCH_START_SEC, SEARCH_END_SEC, BRIGHTNESS_THRESHOLD
        )
    except IOError as e:
        result["notes"] = str(e)
        print(f"  [{pid}] ERROR: {e}")
        return result

    if ref_lo_sec is None:
        result["notes"] = "Lights-on not detected in reference video"
        print(f"  [{pid}] WARNING: {result['notes']}")
        return result

    result["reference_lights_on_sec"] = round(ref_lo_sec, 4)

    # Step 2: find lights-on in the target video
    try:
        tgt_lo_sec, _ = find_lights_on(
            tgt_video, SEARCH_START_SEC, SEARCH_END_SEC, BRIGHTNESS_THRESHOLD
        )
    except IOError as e:
        result["notes"] = str(e)
        print(f"  [{pid}] ERROR: {e}")
        return result

    if tgt_lo_sec is None:
        result["notes"] = "Lights-on not detected in target video"
        print(f"  [{pid}] WARNING: {result['notes']}")
        return result

    result["target_lights_on_sec"] = round(tgt_lo_sec, 4)
    result["offset_sec"] = round(tgt_lo_sec - ref_lo_sec, 4)

    print(f"  [{pid}] offset = {result['offset_sec']:.4f}s")
    return result


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    results = []

    for _, row in df.iterrows():
        print(f"\nProcessing: {row['participant_id']}")
        results.append(process_row(row))

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Results saved to {OUTPUT_CSV}")
