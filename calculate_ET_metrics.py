"""Purpose: Converts fixation sequences into comprehensive behavioral summary statistics (wide-form).

INPUT REQUIREMENTS:
- A CSV file containing fixation events.
- Required columns: 'name' (AOI label), 'start frame', 'end frame', 'frame duration'.

CORE METRICS CALCULATED PER PARTICIPANT & AOI (Table X - Eye-tracking Metrics):
1. Dwell Time: Total ms spent on an AOI (dwell_time)
2. Proportion of Dwell Time: AOI dwell as fraction of total gaze time (proportion_of_dwell_time)
3. Fixation Count: Number of discrete fixations (fixation_count)
4. Fixation Duration: Min/max individual fixation lengths (fixation_duration_min, fixation_duration_max)
5. Mean Fixation Duration: Average fixation length per AOI (mean_fixation_duration_per_aoi)
6. Latency to First Fixation: Time from session onset to first AOI look (latency_to_first_fixation)
7. Transition Probability: Likelihood of gaze shifting between AOIs (transition_probability_to_[target])
8. Temporal Dynamics: Gaze allocation changes across task phases (temporal_dynamics_phase[N])

Wide-form output: One row per participant-AOI pair with all metrics as columns.
Eye-tracking metrics calculator for comprehensive attention analysis.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

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


def calculate_dwell_time(fixation_events_for_aoi: List[Dict[str, Any]]) -> float:
    """Calculate total dwell time (sum of fixation durations) for an AOI."""
    return float(sum(r["duration_ms"] for r in fixation_events_for_aoi))


def calculate_proportion_of_dwell_time(
    dwell_time: float,
    participant_total_time: float
) -> float:
    """Calculate dwell time as proportion of total participant session time."""
    return _safe_div(dwell_time, participant_total_time)


def calculate_fixation_count(fixation_events_for_aoi: List[Dict[str, Any]]) -> int:
    """Count number of discrete fixations for an AOI."""
    return len(fixation_events_for_aoi)


def calculate_fixation_duration_stats(
    fixation_events_for_aoi: List[Dict[str, Any]]
) -> Tuple[float, float]:
    """
    Extract min and max fixation durations for an AOI.
    
    Returns:
        Tuple of (min_duration_ms, max_duration_ms). Returns (0.0, 0.0) if no fixations.
    """
    if not fixation_events_for_aoi:
        return 0.0, 0.0
    
    durations = [r["duration_ms"] for r in fixation_events_for_aoi]
    return float(min(durations)), float(max(durations))


def calculate_mean_fixation_duration_per_aoi(
    fixation_events_for_aoi: List[Dict[str, Any]]
) -> float:
    """Calculate mean duration of fixations for an AOI."""
    if not fixation_events_for_aoi:
        return 0.0
    
    durations = [r["duration_ms"] for r in fixation_events_for_aoi]
    return _safe_div(float(sum(durations)), len(durations))


def calculate_latency_to_first_fixation(
    fixation_events_for_aoi: List[Dict[str, Any]]
) -> float:
    """
    Calculate time from session start to first fixation on AOI.
    
    Returns:
        timestamp_ms of first fixation, or -1.0 if AOI never fixated.
    """
    if not fixation_events_for_aoi:
        return -1.0
    
    events_sorted = sorted(fixation_events_for_aoi, key=lambda x: x["timestamp_ms"])
    return float(events_sorted[0]["timestamp_ms"])


def calculate_transition_probabilities(
    participant_id: str,
    all_participant_events: List[Dict[str, Any]],
    all_aois: set,
) -> Dict[str, Dict[str, int]]:
    """
    Calculate transition matrices from all AOIs to all target AOIs.
    
    Args:
        participant_id: Filter events for this participant
        all_participant_events: All fixation events (will be filtered)
        all_aois: Set of all unique AOI names in session
    
    Returns:
        Dict mapping source_aoi → {target_aoi: transition_count}
    """
    # Filter events for this participant, sort by timestamp
    pid_events = sorted(
        [e for e in all_participant_events if e["participant_id"] == participant_id],
        key=lambda x: x["timestamp_ms"]
    )
    
    # Initialize transition matrix
    transitions_from_all = {aoi: {} for aoi in all_aois}
    
    if len(pid_events) < 2:
        # No transitions possible
        return transitions_from_all
    
    # Build transition counts
    for i in range(len(pid_events) - 1):
        current_aoi = pid_events[i]["aoi_name"]
        next_aoi = pid_events[i + 1]["aoi_name"]
        
        if current_aoi not in transitions_from_all:
            transitions_from_all[current_aoi] = {}
        
        transitions_from_all[current_aoi][next_aoi] = (
            transitions_from_all[current_aoi].get(next_aoi, 0) + 1
        )
    
    return transitions_from_all


def calculate_temporal_dynamics(
    fixation_events_for_aoi: List[Dict[str, Any]],
    min_session_frame: int,
    max_session_frame: int,
    participant_total_time: float,
    num_phases: int = 3,
) -> Dict[int, float]:
    """
    Calculate dwell time proportion within each phase.
    
    Divides session into equal-length phases. If a fixation crosses a phase boundary,
    splits its duration proportionally by frame count.
    
    Args:
        fixation_events_for_aoi: All fixations for this AOI
        min_session_frame: Earliest start_frame in entire session
        max_session_frame: Latest end_frame in entire session
        participant_total_time: Total participant dwell time (ms)
        num_phases: Number of temporal phases (default: 3)
    
    Returns:
        Dict mapping phase (1-indexed) → proportion_of_total_time
    """
    if num_phases < 1:
        num_phases = 3
    
    total_frames = max_session_frame - min_session_frame + 1
    frames_per_phase = total_frames / num_phases
    
    # Initialize dwell time per phase
    phase_dwell_times = {i: 0.0 for i in range(1, num_phases + 1)}
    
    for event in fixation_events_for_aoi:
        start_frame = event["start_frame"]
        end_frame = event["end_frame"]
        duration_ms = event["duration_ms"]
        duration_frames = event["duration_frames"]
        
        # Iterate through each frame in the fixation
        for frame_idx in range(start_frame, end_frame + 1):
            # Which phase does this frame belong to?
            relative_frame = frame_idx - min_session_frame
            phase_num = min(
                int(relative_frame / frames_per_phase) + 1,
                num_phases
            )
            
            # Allocate 1 frame worth of duration to this phase
            frame_duration_ms = duration_ms / duration_frames
            phase_dwell_times[phase_num] += frame_duration_ms
    
    # Convert to proportions
    phase_proportions = {
        phase: _safe_div(dwell, participant_total_time)
        for phase, dwell in phase_dwell_times.items()
    }
    
    return phase_proportions


def calculate_summary_scores(
    fixation_events: List[Dict[str, Any]],
    num_phases: int = 3
) -> List[Dict[str, Any]]:
    """
    Aggregate fixation events into comprehensive summary statistics (wide-form CSV).
    
    Computes all eye-tracking metrics per participant-AOI pair in the order specified:
    1. dwell_time
    2. proportion_of_dwell_time
    3. fixation_count
    4. fixation_duration_min/max
    5. mean_fixation_duration_per_aoi
    6. latency_to_first_fixation
    7. transition_probability_to_[AOIs] (alphabetically sorted targets)
    8. temporal_dynamics_phase[N]
    
    Args:
        fixation_events: List of fixation event dicts
        num_phases: Number of temporal phases (default: 3)
    
    Returns:
        List of wide-form summary dicts (one row per participant-AOI)
    """
    if not fixation_events:
        return []
    
    # ========== SESSION-WIDE STATISTICS ==========
    grouped = {}
    participant_totals = {}
    all_participants = set()
    all_aois = set()
    min_session_frame = float('inf')
    max_session_frame = float('-inf')
    
    # First pass: group by (participant, aoi), calculate totals
    for row in fixation_events:
        pid = row["participant_id"]
        aoi = row["aoi_name"]
        key = (pid, aoi)
        
        grouped.setdefault(key, []).append(row)
        participant_totals[pid] = participant_totals.get(pid, 0.0) + float(row["duration_ms"])
        all_participants.add(pid)
        all_aois.add(aoi)
        
        min_session_frame = min(min_session_frame, row["start_frame"])
        max_session_frame = max(max_session_frame, row["end_frame"])
    
    # Build transition matrices per participant
    participant_transitions = {}
    for pid in all_participants:
        participant_transitions[pid] = calculate_transition_probabilities(
            pid, fixation_events, all_aois
        )
    
    # Sort all_aois alphabetically for consistent column order
    sorted_aois = sorted(all_aois)
    
    summary_rows = []
    
    # ========== PER-PARTICIPANT-PER-AOI CALCULATION ==========
    for (pid, aoi), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        rows_sorted = sorted(rows, key=lambda x: x["timestamp_ms"])
        
        # 1. Dwell Time
        dwell_time = calculate_dwell_time(rows_sorted)
        
        # 2. Proportion of Dwell Time
        proportion_of_dwell_time = calculate_proportion_of_dwell_time(
            dwell_time, participant_totals[pid]
        )
        
        # 3. Fixation Count
        fixation_count = calculate_fixation_count(rows_sorted)
        
        # 4. Fixation Duration Min/Max
        fixation_duration_min, fixation_duration_max = calculate_fixation_duration_stats(
            rows_sorted
        )
        
        # 5. Mean Fixation Duration Per AOI
        mean_fixation_duration_per_aoi = calculate_mean_fixation_duration_per_aoi(
            rows_sorted
        )
        
        # 6. Latency to First Fixation
        latency_to_first_fixation = calculate_latency_to_first_fixation(rows_sorted)
        
        # 7. Transition Probabilities FROM this AOI to all targets
        transition_dict = participant_transitions[pid][aoi]
        transition_probs = {}
        for target_aoi in sorted_aois:
            count_to_target = transition_dict.get(target_aoi, 0)
            total_transitions = sum(transition_dict.values())
            prob = _safe_div(count_to_target, total_transitions)
            transition_probs[f"transition_probability_to_{target_aoi}"] = prob
        
        # 8. Temporal Dynamics (phases)
        phase_proportions = calculate_temporal_dynamics(
            rows_sorted,
            min_session_frame,
            max_session_frame,
            participant_totals[pid],
            num_phases=num_phases
        )
        
        # ========== ASSEMBLE WIDE-FORM ROW ==========
        row_dict = {
            "participant_id": pid,
            "aoi_name": aoi,
            "dwell_time": dwell_time,
            "proportion_of_dwell_time": proportion_of_dwell_time,
            "fixation_count": fixation_count,
            "fixation_duration_min": fixation_duration_min,
            "fixation_duration_max": fixation_duration_max,
            "mean_fixation_duration_per_aoi": mean_fixation_duration_per_aoi,
            "latency_to_first_fixation": latency_to_first_fixation,
        }
        
        # Add transition probabilities (alphabetically sorted by target)
        row_dict.update(transition_probs)
        
        # Add temporal dynamics phases
        for phase_num in range(1, num_phases + 1):
            row_dict[f"temporal_dynamics_phase{phase_num}"] = (
                phase_proportions.get(phase_num, 0.0)
            )
        
        summary_rows.append(row_dict)
    
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
    parser.add_argument(
        "--num-phases",
        type=int,
        default=3,
        help="Number of temporal phases to divide session into (default: 3)",
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
    summary_scores = calculate_summary_scores(fixation_events, num_phases=args.num_phases)
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
              f"{score['fixation_count']} fixations, "
              f"{score['dwell_time']:.1f}ms dwell, "
              f"latency={score['latency_to_first_fixation']:.1f}ms")
