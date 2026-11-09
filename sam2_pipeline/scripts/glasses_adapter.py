"""Gaze data adapter for METMAP.

Converts tabular gaze exports from glassesTools (Niehorster et al., 2023),
Pupil Labs (Core/Invisible/Neon), Tobii Pro Glasses 2/3, or custom CSV/TSV
files into METMAP's standardized format:
    ['frame_idx', 'gaze_x_norm', 'gaze_y_norm', 'timestamp_sec']
with bottom-left normalized coordinates and timestamps in seconds.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_universal_gaze(
    gaze_path,
    format_type="auto",
    video_width=None,
    video_height=None,
    *,
    coordinate_origin=None,
    coordinate_units=None,
    timestamp_unit=None,
    frame_column=None,
    x_column=None,
    y_column=None,
    timestamp_column=None,
) -> pd.DataFrame:
    path = Path(gaze_path)
    if not path.is_file():
        raise FileNotFoundError(f"Gaze file not found: {path}")

    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    if df.empty or df.columns.duplicated().any():
        raise ValueError("Gaze export must contain non-empty rows and unique columns.")

    cols = set(df.columns)
    valid_formats = ("auto", "pupil_labs", "glassestools", "tobii", "standardized", "generic")
    if format_type not in valid_formats:
        raise ValueError(f"Unsupported format_type '{format_type}'. Choose from {valid_formats}.")

    if format_type == "auto":
        if {"world_index", "norm_pos_x", "norm_pos_y"} <= cols:
            format_type = "pupil_labs"
        elif {"frame_idx", "gaze_x_norm", "gaze_y_norm"} <= cols:
            format_type = "standardized"
        elif {"frame_idx", "gaze_x", "gaze_y"} <= cols:
            format_type = "glassestools"
        elif {"Frame number", "Gaze point 2D X", "Gaze point 2D Y"} <= cols:
            format_type = "tobii"
        elif all((frame_column, x_column, y_column)):
            format_type = "generic"
        else:
            raise ValueError(
                "Could not auto-detect gaze schema. Specify --frame-column, --x-column, "
                "and --y-column along with --coordinate-units and --coordinate-origin."
            )

    if format_type == "pupil_labs":
        fc, xc, yc = frame_column or "world_index", x_column or "norm_pos_x", y_column or "norm_pos_y"
        units, origin = coordinate_units or "normalized", coordinate_origin or "bottom_left"
    elif format_type in ("standardized", "glassestools"):
        fc = frame_column or "frame_idx"
        is_norm = {"gaze_x_norm", "gaze_y_norm"} <= cols
        xc = x_column or ("gaze_x_norm" if is_norm else "gaze_x")
        yc = y_column or ("gaze_y_norm" if is_norm else "gaze_y")
        units = coordinate_units or ("normalized" if is_norm else "pixels")
        origin = coordinate_origin or ("bottom_left" if is_norm else "top_left")
    elif format_type == "tobii":
        fc = frame_column or "Frame number"
        xc = x_column or "Gaze point 2D X"
        yc = y_column or "Gaze point 2D Y"
        units = coordinate_units or "normalized"
        origin = coordinate_origin or "top_left"
    else:
        fc, xc, yc = frame_column, x_column, y_column
        units, origin = coordinate_units, coordinate_origin

    if not all(c in cols for c in (fc, xc, yc)):
        raise ValueError(f"Required gaze columns missing: {fc}, {xc}, {yc}")
    if units not in ("normalized", "pixels") or origin not in ("bottom_left", "top_left"):
        raise ValueError("Specify coordinate_units ('normalized' or 'pixels') and coordinate_origin ('bottom_left' or 'top_left').")

    frames = pd.to_numeric(df[fc], errors="raise")
    if not np.isfinite(frames).all() or (frames < 0).any() or (frames % 1 != 0).any():
        raise ValueError("Frame indices must be finite non-negative integers.")

    x = pd.to_numeric(df[xc], errors="raise").astype(np.float64)
    y = pd.to_numeric(df[yc], errors="raise").astype(np.float64)

    if units == "pixels":
        if any(v is None or not np.isfinite(v) or v <= 0 for v in (video_width, video_height)):
            raise ValueError("Pixel coordinates require positive --video-width and --video-height.")
        x = x / float(video_width)
        y = y / float(video_height)

    if origin == "top_left":
        y = 1.0 - y

    tc = timestamp_column
    if tc is None:
        tc = next(
            (c for c in ("gaze_timestamp", "world_timestamp", "timestamp_sec", "timestamp", "Recording timestamp", "time") if c in cols),
            None,
        )
    if tc is not None and tc not in cols:
        raise ValueError(f"Timestamp column not found: {tc}")

    unit = timestamp_unit
    if unit is None:
        if tc == "Recording timestamp":
            unit = "ms"
        elif tc in ("gaze_timestamp", "world_timestamp", "timestamp_sec", "timestamp", "time"):
            unit = "s"
    if tc is not None and unit not in ("s", "ms", "us", "ns"):
        raise ValueError("Specify --timestamp-unit ('s', 'ms', 'us', or 'ns').")

    times = pd.to_numeric(df[tc], errors="raise").astype(np.float64) if tc else pd.Series(np.nan, index=df.index)
    scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}.get(unit, 1.0)

    return pd.DataFrame({
        "frame_idx": frames.astype(np.int64),
        "gaze_x_norm": x,
        "gaze_y_norm": y,
        "timestamp_sec": times * scale,
    })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to source gaze CSV or TSV file")
    parser.add_argument("--format", default="auto", choices=["auto", "pupil_labs", "glassestools", "tobii", "standardized", "generic"])
    parser.add_argument("--video-width", type=int, help="Video width in pixels (required when input coordinates are in pixels)")
    parser.add_argument("--video-height", type=int, help="Video height in pixels (required when input coordinates are in pixels)")
    parser.add_argument("--coordinate-origin", choices=["bottom_left", "top_left"])
    parser.add_argument("--coordinate-units", choices=["normalized", "pixels"])
    parser.add_argument("--timestamp-unit", choices=["s", "ms", "us", "ns"])
    for col in ("frame", "x", "y", "timestamp"):
        parser.add_argument(f"--{col}-column")
    parser.add_argument("--output", required=True, help="Output path for standardized CSV")
    args = parser.parse_args(argv)

    data = load_universal_gaze(
        args.input,
        args.format,
        args.video_width,
        args.video_height,
        coordinate_origin=args.coordinate_origin,
        coordinate_units=args.coordinate_units,
        timestamp_unit=args.timestamp_unit,
        frame_column=args.frame_column,
        x_column=args.x_column,
        y_column=args.y_column,
        timestamp_column=args.timestamp_column,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    print(f"Standardized {len(data)} gaze rows -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
