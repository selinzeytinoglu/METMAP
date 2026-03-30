"""
Purpose: Converts fixation sequences into behavioral summary statistics.

INPUT REQUIREMENTS:
- A CSV file containing fixation events.
- Required columns: 'name' (AOI label), 'start frame', 'end frame', 'frame duration'.

CORE METRICS CALCULATED PER PARTICIPANT & AOI:
1. Dwell Time: Total ms spent on an AOI (total_dwell_time_ms)
2. Fixation Count: Number of distinct looks (num_fixations)
3. Avg Duration: Mean length of each fixation (avg_fixation_duration_ms)
4. Dwell Proportion: AOI dwell time as fraction of total session time (dwell_proportion)

Eye-tracking metrics calculator.
Converts fixation data into comprehensive AOI metrics including dwell time,
fixation counts, and temporal statistics.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def _safe_div(numerator: float, denominator: float) -> float:
    """Safely divide, returning 0.0 if denominator is zero."""
    return float(numerator) / float(denominator) if denominator else 0.0


def calculate_fixation_events(
    dominant_sequence: List[str],
    fps: float,
    participant_id: str = "TEST_001",
    include_none: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convert a sequence of dominant AOI labels into discrete fixation events.
    
    Groups consecutive identical labels into single fixation periods, computing
    duration in both millisecond and frame counts, plus temporal metadata.
    
    Args:
        dominant_sequence: List of AOI labels (e.g., ['Mother', 'Mother', 'Child', ...])
        fps: Frames per second for timestamp conversion
        participant_id: Participant identifier (default: "TEST_001")
        include_none: Whether to include "None" fixations (default: False)
    
    Returns:
        List of dicts with keys: participant_id, aoi_name, duration_ms, duration_frames,
        timestamp_ms, start_frame, end_frame
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")

    frame_ms = 1000.0 / float(fps)
    events = []

    if not dominant_sequence:
        return events

    run_label = dominant_sequence[0]
    run_start = 0

    for idx, label in enumerate(dominant_sequence[1:], start=1):
        if label != run_label:
            run_end = idx - 1
            run_len = run_end - run_start + 1

            if include_none or run_label != "None":
                events.append(
                    {
                        "participant_id": participant_id,
                        "aoi_name": run_label,
                        "duration_ms": run_len * frame_ms,
                        "duration_frames": run_len,
                        "timestamp_ms": run_start * frame_ms,
                        "start_frame": run_start,
                        "end_frame": run_end,
                    }
                )

            run_label = label
            run_start = idx

    run_end = len(dominant_sequence) - 1
    run_len = run_end - run_start + 1
    if include_none or run_label != "None":
        events.append(
            {
                "participant_id": participant_id,
                "aoi_name": run_label,
                "duration_ms": run_len * frame_ms,
                "duration_frames": run_len,
                "timestamp_ms": run_start * frame_ms,
                "start_frame": run_start,
                "end_frame": run_end,
            }
        )

    return events


def calculate_summary_scores(fixation_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate fixation events into comprehensive summary statistics by participant and AOI.
    
    Computes per AOI:
    - num_fixations: Count of distinct fixations
    - total_dwell_time_ms: Sum of all fixation durations
    - avg_fixation_duration_ms: Mean duration across fixations
    - dwell_proportion: AOI dwell time as fraction of total participant time
    
    Args:
        fixation_events: List of fixation event dicts
    
    Returns:
        List of summary dicts with per-participant-per-AOI statistics
    """
    # Group fixation events by participant and AOI.
    grouped = {}
    participant_totals = {}

    for row in fixation_events:
        pid = row["participant_id"]
        aoi = row["aoi_name"]
        key = (pid, aoi)

        grouped.setdefault(key, []).append(row)
        participant_totals[pid] = participant_totals.get(pid, 0.0) + float(row["duration_ms"])

    summary_rows = []

    # Generate one summary row per (participant, AOI) pair
    for (pid, aoi), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        rows_sorted = sorted(rows, key=lambda x: x["timestamp_ms"])
        durations = [float(r["duration_ms"]) for r in rows_sorted]

        total_dwell = float(sum(durations))
        num_fixations = len(durations)
        
        # Calculate average duration
        avg_duration = _safe_div(total_dwell, num_fixations)

        summary_rows.append(
            {
                "participant_id": pid,
                "aoi_name": aoi,
                "num_fixations": num_fixations,
                "total_dwell_time_ms": total_dwell,
                "avg_fixation_duration_ms": avg_duration,
                "dwell_proportion": _safe_div(total_dwell, participant_totals.get(pid, 0.0)),
            }
        )

    return summary_rows


def load_fixations_from_csv(csv_path: Path, fps: float = 30.0, participant_id: str = None) -> List[Dict[str, Any]]:
    """
    Load fixation data from a CSV file with format:
    name, frame duration, start frame, end frame, duration (s), start (s), end (s)
    
    Converts to fixation event format compatible with calculate_summary_scores.
    
    Args:
        csv_path: Path to fixation CSV
        fps: Frames per second (default: 30.0)
        participant_id: Participant ID (derived from filename if not provided)
    
    Returns:
        List of fixation event dicts
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    if participant_id is None:
        participant_id = csv_path.stem
    
    events = []
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        aoi_name = str(row["name"]).strip()
        start_frame = int(row["start frame"])
        end_frame = int(row["end frame"])
        duration_frames = int(row["frame duration"])
        
        # Convert to milliseconds
        frame_ms = 1000.0 / float(fps)
        duration_ms = duration_frames * frame_ms
        timestamp_ms = start_frame * frame_ms
        
        events.append({
            "participant_id": participant_id,
            "aoi_name": aoi_name,
            "duration_ms": duration_ms,
            "duration_frames": duration_frames,
            "timestamp_ms": timestamp_ms,
            "start_frame": start_frame,
            "end_frame": end_frame,
        })
    
    return events


def save_summary_scores_csv(summary_scores: List[Dict[str, Any]], out_path: Path):
    """Save summary scores to CSV file."""
    if not summary_scores:
        print(f"No summary scores to save to {out_path}")
        return out_path
    
    headers = list(summary_scores[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in summary_scores:
            writer.writerow(row)
    return out_path


def save_summary_scores_json(summary_scores: List[Dict[str, Any]], out_path: Path):
    """Save summary scores to JSON file."""
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary_scores, f, indent=2)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate eye-tracking metrics from fixation CSV data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to fixation CSV file (e.g., 605_fixations.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Base output directory (default: current directory). Results saved to: output_dir/ET_Metrics/{participant_id}/",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second (default: 30.0)",
    )
    parser.add_argument(
        "--participant-id",
        type=str,
        default=None,
        help="Participant ID folder name (default: input filename stem)",
    )
    
    args = parser.parse_args()
    
    # Determine participant ID
    if args.participant_id is None:
        args.participant_id = args.input.stem
    
    # Create nested output directory structure: ET_Metrics/{participant_id}/
    et_metrics_dir = args.output_dir / "ET_Metrics" / args.participant_id
    et_metrics_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading fixations from: {args.input}")
    fixation_events = load_fixations_from_csv(
        args.input,
        fps=args.fps,
        participant_id=args.participant_id,
    )
    print(f"Loaded {len(fixation_events)} fixation events")
    
    print("Calculating summary scores...")
    summary_scores = calculate_summary_scores(fixation_events)
    print(f"Generated {len(summary_scores)} summary score rows")
    
    # Save results to nested structure
    csv_out = et_metrics_dir / f"{args.input.stem}_summary_scores.csv"
    json_out = et_metrics_dir / f"{args.input.stem}_summary_scores.json"
    
    save_summary_scores_csv(summary_scores, csv_out)
    print(f"Saved CSV: {csv_out}")
    
    save_summary_scores_json(summary_scores, json_out)
    print(f"Saved JSON: {json_out}")
    
    print("\nSample summary scores:")
    for score in summary_scores[:5]:
        print(f"  {score['participant_id']} → {score['aoi_name']}: "
              f"{score['num_fixations']} fixations, "
              f"{score['total_dwell_time_ms']:.1f}ms dwell")
