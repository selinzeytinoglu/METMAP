#!/usr/bin/env python3
"""
SAM2 Video Segmentation Batch Processor

This orchestration framework automates high-throughput processing of behavioral 
video datasets. It manages frame extraction, configuration generation, and 
containerized SAM2 inference across multiple participants.

Features:
  - Processes cohorts of any size (N > 100) from a single configuration
  - Fault-tolerant: continues processing if individual participants fail
  - Memory-safe: Docker containers reset between participants, preventing VRAM leaks
  - Comprehensive logging and final summary table showing pass/fail per participant
  - Supports both sequential participant ranges and specific participant lists

Usage:
    python batch_processor.py [--config batch_config.yaml]
"""

import os
import sys
import yaml
import subprocess
import json
import logging
import zipfile
from pathlib import Path, PurePosixPath
from typing import List, Dict, Tuple, Optional
import pandas as pd
from datetime import datetime
from tabulate import tabulate

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

SEGMENTATION_CONFIG_KEYS = {
    "mask_storage",
    "compress_masks",
    "save_masks",
    "write_overlay_video",
    "video_encoder",
    "compile_image_encoder",
    "async_loading_frames",
    "offload_video_to_cpu",
    "offload_state_to_cpu",
    "frame_queue_size",
    "result_queue_size",
    "sam2_frame_prefetch",
    "sam2_frame_prefetch_lookahead",
    "sam2_frame_prefetch_retain",
    "sam2_model_config",
    "sam2_checkpoint",
}


# ============================================================================
# BATCH PROCESSOR ENGINE
# ============================================================================

class BatchProcessor:
    """
    Orchestrates high-throughput SAM2 video segmentation across participant cohorts.
    
    Responsibilities:
      - Load and validate batch configuration
      - Parse participant ID selections (range or list mode)
      - Extract frames locally via FFmpeg
      - Generate per-participant configuration files
      - Launch Docker containers with proper volume mounts
      - Track success/failure for comprehensive reporting
      - Provide fault tolerance: continue on individual participant failures
    """

    def __init__(self, config_path: str = "batch_config.yaml"):
        """
        Initialize batch processor.
        
        Args:
            config_path: Path to batch_config.yaml (default: current directory)
        """
        self.config_path = Path(config_path)
        self.base_dir = Path(__file__).parent
        self.configs_dir = self.base_dir / "configs"
        self.configs_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self._load_config()
        
        # Setup logging
        self._setup_logging()
        
        # Parse participant IDs
        self.participant_ids = self._parse_participant_ids()
        
        # Track results for final summary
        self.results = {
            "successful": [],
            "failed": [],
            "skipped": []
        }

    def _load_config(self) -> None:
        """Load and validate batch configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path.absolute()}\n"
                f"Expected location: {self.base_dir / 'batch_config.yaml'}"
            )
        
        try:
            with open(self.config_path) as f:
                self.cfg = yaml.safe_load(f)
            
            logging.info(f"Loaded batch configuration: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {self.config_path}: {e}")

    def _setup_logging(self) -> None:
        """Configure logging to console and file."""
        log_file = self.base_dir / self.cfg["logging"]["log_file"]
        verbosity = self.cfg["logging"]["verbosity"]
        
        # Determine log level
        level_map = {"quiet": logging.ERROR, "normal": logging.INFO, "verbose": logging.DEBUG}
        log_level = level_map.get(verbosity, logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        console_handler.setFormatter(console_formatter)
        
        # File handler (always DEBUG for comprehensive log)
        if self.cfg["logging"]["save_logs"]:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
            file_handler.setFormatter(file_formatter)
            logging.getLogger().addHandler(file_handler)
        
        # Root logger configuration
        logging.getLogger().setLevel(log_level)
        logging.getLogger().addHandler(console_handler)
        
        logging.info(f"[INIT] Batch processing started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def _parse_participant_ids(self) -> List[str]:
        """
        Extract participant IDs from configuration.
        
        Returns:
            List of zero-padded participant ID strings (e.g., "510", "511")
        """
        mode = self.cfg["batch"]["mode"]
        
        if mode == "list":
            ids = self.cfg["batch"]["participant_ids_list"]
            logging.info(f"[CONFIG] Mode: LIST | Processing {len(ids)} specific participants")
        elif mode == "range":
            start, end = self.cfg["batch"]["participant_ids_range"]
            ids = list(range(start, end + 1))
            logging.info(f"[CONFIG] Mode: RANGE [{start}-{end}] | Processing {len(ids)} participants")
        else:
            raise ValueError(f"Unknown batch mode: {mode}")
        
        return [f"{id:03d}" for id in ids]

    def _load_metadata_csv(self) -> pd.DataFrame:
        """
        Load frame range metadata from CSV.
        
        Expected columns: Participant_ID (or ID), Start_Frame, End_Frame
        
        Returns:
            DataFrame with frame range information
            
        Raises:
            FileNotFoundError: If CSV file specified in config is missing
        """
        csv_path = self.base_dir / self.cfg["frame_extraction"]["metadata_csv_path"]
        
        if not csv_path.exists():
            error_msg = (
                f"Metadata CSV not found: {csv_path.absolute()}\n"
                f"Expected file: {self.cfg['frame_extraction']['metadata_csv_path']}\n"
                f"Create CSV with columns: Participant_ID, Start_Frame, End_Frame"
            )
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            df = pd.read_csv(csv_path)
            
            # Normalize column names (handle both "Participant_ID" and "ID")
            if "ID" in df.columns and "Participant_ID" not in df.columns:
                df.rename(columns={"ID": "Participant_ID"}, inplace=True)
            
            logging.debug(f"Loaded metadata from {csv_path.name}: {len(df)} rows")
            return df
        except Exception as e:
            logging.error(f"Failed to parse CSV {csv_path}: {e}")
            raise

    @staticmethod
    def _expected_frame_count(start_frame: int, end_frame: int) -> int:
        """Return inclusive frame count for metadata Start_Frame/End_Frame."""
        count = int(end_frame) - int(start_frame) + 1
        if count <= 0:
            raise ValueError(f"Invalid frame range: {start_frame}-{end_frame}")
        return count

    @staticmethod
    def _frame_file_index(path: Path) -> Optional[int]:
        try:
            return int(path.stem)
        except ValueError:
            return None

    def _list_frame_files(self, frames_dir: Path) -> List[Path]:
        if not frames_dir.exists():
            return []
        files = [
            path for path in frames_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        ]
        return sorted(files, key=lambda path: (self._frame_file_index(path) is None, self._frame_file_index(path) or 0, path.name))

    def _validate_frame_folder(self, frames_dir: Path, expected_count: int, context: str) -> bool:
        frame_files = self._list_frame_files(frames_dir)
        if len(frame_files) != expected_count:
            logging.warning(
                f"[{context}] Frame count mismatch in {frames_dir}: "
                f"expected {expected_count}, found {len(frame_files)}"
            )
            return False

        indexes = [self._frame_file_index(path) for path in frame_files]
        if any(index is None for index in indexes):
            bad = [path.name for path, index in zip(frame_files, indexes) if index is None][:10]
            logging.warning(f"[{context}] Non-numeric frame filenames in {frames_dir}: {bad}")
            return False

        expected_indexes = set(range(expected_count))
        actual_indexes = set(indexes)
        missing = sorted(expected_indexes - actual_indexes)
        unexpected = sorted(actual_indexes - expected_indexes)
        duplicate_count = len(indexes) - len(actual_indexes)
        if missing or unexpected or duplicate_count:
            logging.warning(
                f"[{context}] Frame index mismatch in {frames_dir}: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}, duplicates={duplicate_count}"
            )
            return False

        return True

    def _clear_frame_files(self, frames_dir: Path) -> None:
        for frame_file in self._list_frame_files(frames_dir):
            frame_file.unlink()

    def _segmentation_script(self) -> str:
        segmentation_cfg = self.cfg.get("segmentation", {}) or {}
        script = str(segmentation_cfg.get("script", "scripts/segment_video.py")).strip().replace("\\", "/")
        script_path = PurePosixPath(script)
        if (
            not script
            or script_path.is_absolute()
            or ".." in script_path.parts
            or script_path.parts[:1] != ("scripts",)
            or script_path.suffix != ".py"
        ):
            raise ValueError(
                "segmentation.script must be a relative Python script under scripts/, "
                f"got {script!r}"
            )
        return script

    def _segmentation_config_for_participant(self) -> Dict:
        segmentation_cfg = self.cfg.get("segmentation", {}) or {}
        participant_settings = {}
        for key in SEGMENTATION_CONFIG_KEYS:
            if key not in segmentation_cfg:
                continue
            value = segmentation_cfg[key]
            if isinstance(value, str) and value.strip() == "":
                continue
            participant_settings[key] = value
        return participant_settings

    def _participant_output_dir(self, participant_id: str) -> Path:
        suffix = self.cfg["frame_extraction"]["suffix"]
        return Path(self.cfg["output"]["base_output_dir"]) / f"{participant_id}_{suffix}_output"

    @staticmethod
    def _mask_file_index(path: Path) -> Optional[int]:
        stem = path.stem
        if stem.startswith("frame_"):
            stem = stem[len("frame_"):]
        try:
            return int(stem)
        except ValueError:
            return None

    def _validate_segmentation_outputs(self, participant_id: str, expected_count: int) -> bool:
        output_dir = self._participant_output_dir(participant_id)
        masks_dir = output_dir / "masks"
        segmentation_cfg = self.cfg.get("segmentation", {}) or {}
        save_masks = bool(segmentation_cfg.get("save_masks", True))
        write_overlay_video = bool(segmentation_cfg.get("write_overlay_video", True))
        script = self._segmentation_script()
        ok = True

        logging.info(f"[VALIDATE] Checking segmentation outputs for {participant_id}")

        if save_masks:
            mask_files = sorted(
                masks_dir.glob("frame_*.npz") if masks_dir.exists() else [],
                key=lambda path: (self._mask_file_index(path) is None, self._mask_file_index(path) or 0, path.name),
            )
            if len(mask_files) != expected_count:
                logging.error(
                    f"[VALIDATE] Mask count mismatch for {participant_id}: "
                    f"expected {expected_count}, found {len(mask_files)}"
                )
                ok = False
            else:
                indexes = [self._mask_file_index(path) for path in mask_files]
                expected_indexes = set(range(expected_count))
                actual_indexes = set(indexes)
                missing = sorted(expected_indexes - actual_indexes)
                unexpected = sorted(actual_indexes - expected_indexes)
                duplicate_count = len(indexes) - len(actual_indexes)
                if missing or unexpected or duplicate_count:
                    logging.error(
                        f"[VALIDATE] Mask index mismatch for {participant_id}: "
                        f"missing={missing[:10]}, unexpected={unexpected[:10]}, duplicates={duplicate_count}"
                    )
                    ok = False
                else:
                    logging.info(f"[VALIDATE] Masks: {len(mask_files)} file(s) aligned with frames")

                if mask_files:
                    try:
                        with zipfile.ZipFile(mask_files[0]) as npz_file:
                            labels = sorted(
                                Path(name).stem
                                for name in npz_file.namelist()
                                if name.endswith(".npy") and not Path(name).stem.startswith("__")
                            )
                        logging.info(f"[VALIDATE] Sample mask labels: {labels}")
                    except Exception as exc:
                        logging.warning(f"[VALIDATE] Could not inspect sample mask labels: {exc}")
        else:
            logging.info("[VALIDATE] Mask saving disabled; skipping mask count check")

        output_video = output_dir / "output_video.mp4"
        if write_overlay_video:
            if output_video.exists() and output_video.stat().st_size > 0:
                logging.info(f"[VALIDATE] Overlay video exists: {output_video.name}")
            else:
                logging.error(f"[VALIDATE] Missing overlay video for {participant_id}: {output_video}")
                ok = False
        else:
            logging.info("[VALIDATE] Overlay video disabled; skipping video check")

        timing_path = output_dir / "timing_report_even_faster.json"
        if script.endswith("segment_video_even_faster.py"):
            if timing_path.exists():
                try:
                    with open(timing_path, encoding="utf-8") as f:
                        timing = json.load(f)
                    logging.info(
                        "[VALIDATE] Timing: "
                        f"{timing.get('frames', '->')} frame(s), "
                        f"{timing.get('total_seconds', '->')}s total, "
                        f"{timing.get('end_to_end_fps', '->')} end-to-end FPS"
                    )
                except Exception as exc:
                    logging.warning(f"[VALIDATE] Could not read timing report: {exc}")
            else:
                logging.warning(f"[VALIDATE] Timing report missing: {timing_path}")

        return ok

    def _extract_frames(self, participant_id: str, start_frame: int, end_frame: int) -> Tuple[bool, Optional[str]]:
        """
        Extract video frames locally using FFmpeg.
        
        Extracts a frame range from source video file using FFmpeg. Supports both:
          - Direct extraction from raw video files (if video_path_pattern specified)
          - Reusing existing extracted frames (if already present on disk)
        
        Args:
            participant_id: Participant ID (e.g., "510")
            start_frame: First frame to extract
            end_frame: Last frame to extract
            
        Returns:
            (success: bool, frames_dir: str or None)
        """
        suffix = self.cfg["frame_extraction"]["suffix"]
        frames_dir = Path(self.cfg["paths"]["local_data_dir"]) / f"{participant_id}_{suffix}_frames"
        expected_count = self._expected_frame_count(start_frame, end_frame)
        extraction_enabled = bool(self.cfg["frame_extraction"]["enabled"])

        existing_frames = self._list_frame_files(frames_dir)
        if existing_frames:
            if self._validate_frame_folder(frames_dir, expected_count, "EXTRACT"):
                logging.info(f"[EXTRACT] Reusing complete frame folder: {frames_dir.name}")
                return True, str(frames_dir)
            if not extraction_enabled:
                logging.error(
                    f"[EXTRACT] Existing frame folder is incomplete and extraction is disabled: {frames_dir}"
                )
                return False, None
            logging.warning(f"[EXTRACT] Removing incomplete frame files before re-extraction: {frames_dir}")
            self._clear_frame_files(frames_dir)

        if not extraction_enabled:
            logging.warning(f"[EXTRACT] Frame extraction disabled and no complete pre-extracted frames found for {participant_id}")
            return False, None

        logging.info(f"[EXTRACT] Participant {participant_id}: extracting frames {start_frame}-{end_frame}")
        
        # Get video source path from config
        video_path_pattern = self.cfg["frame_extraction"].get("video_path_pattern", "")
        
        if not video_path_pattern or video_path_pattern.strip() == "":
            logging.error(
                f"[EXTRACT] No video_path_pattern configured and no complete frame folder found for {participant_id}"
            )
            return False, None
        
        # Construct actual video path
        video_path = video_path_pattern.replace("{id}", participant_id)
        video_path_obj = Path(video_path)
        
        if not video_path_obj.exists():
            logging.warning(f"[EXTRACT] Source video not found for {participant_id}: {video_path}")
            logging.warning(f"[EXTRACT] Skipping frame extraction; ensure frames exist or verify video_path_pattern")
            return False, None
        
        # Construct FFmpeg command for frame extraction
        try:
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # FFmpeg filter: select only frames between start_frame and end_frame (inclusive)
            filter_expr = f"select='between(n\\,{int(start_frame)}\\,{int(end_frame)})'"
            
            # FFmpeg command parameters:
            # -y: overwrite existing output without asking
            # -threads 1: single-threaded for stability
            # -i: input video file
            # -vf: video filter (frame selection)
            # -vsync 0: no frame sync (preserve exact frame count)
            # -q:v 2: high quality JPEG output (2=highest quality, 31=lowest)
            # -start_number 0: start naming at 00000.jpg
            # -loglevel error: suppress verbose FFmpeg output
            command = [
                "ffmpeg", "-y", "-threads", "1",
                "-i", str(video_path),
                "-vf", filter_expr,
                "-vsync", "0",
                "-q:v", "2",
                "-start_number", "0",
                "-loglevel", "error",
                os.path.join(str(frames_dir), "%05d.jpg")
            ]
            
            logging.debug(f"[EXTRACT] FFmpeg command: {' '.join(command)}")
            
            # Execute FFmpeg extraction
            result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                extracted_count = len(self._list_frame_files(frames_dir))
                if not self._validate_frame_folder(frames_dir, expected_count, "EXTRACT"):
                    logging.error(f"[EXTRACT] FFmpeg output failed validation for {participant_id}")
                    return False, None
                logging.info(f"[EXTRACT] Successfully extracted {extracted_count} frames to {frames_dir.name}")
                return True, str(frames_dir)
            else:
                logging.error(f"[EXTRACT] FFmpeg failed for {participant_id} (exit code: {result.returncode})")
                if result.stderr:
                    logging.debug(f"[EXTRACT] FFmpeg stderr: {result.stderr}")
                return False, None
                
        except subprocess.TimeoutExpired:
            logging.error(f"[EXTRACT] FFmpeg timeout for {participant_id} (exceeded 1 hour)")
            return False, None
        except Exception as e:
            logging.error(f"[EXTRACT] Unexpected error during frame extraction for {participant_id}: {e}")
            return False, None

    def _generate_participant_config(self, participant_id: str, start_frame: int, end_frame: int, 
                                     frames_dir: str) -> Tuple[bool, Optional[Path]]:
        """
        Generate per-participant configuration file for Docker container.
        
        This dynamically creates a config.yaml that segment_video.py will read.
        Handles file mapping: host {id}_coordinates.json -> Docker coordinates.json
        
        Args:
            participant_id: Participant ID
            start_frame: First frame index
            end_frame: Last frame index
            frames_dir: Path to extracted frames
            
        Returns:
            (success: bool, config_file_path: Path or None)
        """
        try:
            suffix = self.cfg["frame_extraction"]["suffix"]
            
            expected_count = self._expected_frame_count(start_frame, end_frame)
            if not self._validate_frame_folder(Path(frames_dir), expected_count, "CONFIG"):
                raise ValueError(f"Frame folder failed validation: {frames_dir}")

            # Generate per-participant configuration
            participant_cfg = {
                "participant_id": int(participant_id),
                "fps": self.cfg["defaults"]["fps"],
                "start_frame": 0,  # Frames already extracted, relative indexing
                "end_frame": expected_count - 1,
                
                # Docker paths (IMPORTANT: must match VOLUME in Dockerfile)
                "local_data_dir": self.cfg["paths"]["local_data_dir"],
                "docker_data_dir": self.cfg["paths"]["docker_data_dir"],
                
                # Relative paths inside docker_data_dir
                "frames_dir": f"{participant_id}_{suffix}_frames",
                "prompt_coordinates": self.cfg["prompts"]["docker_filename"],
                "masks_dir": "output/masks",
                "output_dir": "output",
                
                # ROI labels
                "labels": self.cfg["defaults"]["labels"],
                "label_profiles": self.cfg["defaults"].get("label_profiles", {}),
                
                # Gaze mapping
                "uncertainty_radius": self.cfg["gaze_mapping"]["uncertainty_radius"]
            }
            participant_cfg.update(self._segmentation_config_for_participant())
            
            # Write participant config
            config_file = self.configs_dir / f"participant_{participant_id}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(participant_cfg, f, default_flow_style=False, sort_keys=False)
            
            logging.debug(f"[CONFIG] Generated: {config_file.name}")
            return True, config_file
        except Exception as e:
            logging.error(f"[ERROR] Failed to generate config for {participant_id}: {e}")
            return False, None

    def _validate_prompt_file(self, participant_id: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that prompt file exists on the host.
        
        Args:
            participant_id: Participant ID
            
        Returns:
            (exists: bool, prompt_file_path: str or None)
        """
        base_dir = Path(self.cfg["prompts"]["base_dir"])
        pattern = self.cfg["prompts"]["file_pattern"].replace("{id}", participant_id)
        prompt_file = base_dir / pattern
        
        if not prompt_file.exists():
            logging.warning(f"[PROMPT] Not found for {participant_id}: {prompt_file.name}")
            return False, None
        
        logging.debug(f"[PROMPT] Found: {prompt_file.name}")
        return True, str(prompt_file)

    def _run_docker_segmentation(self, participant_id: str, frames_dir: str, 
                                  prompt_file: str, config_file: Path) -> bool:
        """
        Launch Docker container for SAM2 segmentation.
        
        Constructs docker run command with proper volume mounts and executes.
        
        Args:
            participant_id: Participant ID
            frames_dir: Host path to frames
            prompt_file: Host path to prompt coordinates
            config_file: Generated per-participant config
            
        Returns:
            success: True if Docker container executed successfully
        """
        try:
            docker_data_dir = self.cfg["paths"]["docker_data_dir"]
            output_dir = self._participant_output_dir(participant_id)
            segmentation_script = self._segmentation_script()
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Construct docker run command
            docker_cmd = [
                "docker", "run",
                "--gpus", self.cfg["docker"]["gpus"],
                "--rm",  # Clean up container after run
                
                # Mount data directory (frames)
                "-v", f"{frames_dir}:{docker_data_dir}",
                
                # Mount prompt coordinates  
                "-v", f"{prompt_file}:{docker_data_dir}/{self.cfg['prompts']['docker_filename']}",
                
                # Mount config file
                "-v", f"{config_file}:/config/config.yaml",
                
                # Mount output directory
                "-v", f"{output_dir}:{docker_data_dir}/output",
                
                # Image and command
                self.cfg["docker"]["image_name"],
                "python", segmentation_script, participant_id
            ]
            
            logging.info(f"[DOCKER] Launching container for {participant_id}")
            logging.debug(f"[DOCKER] Command: {' '.join(docker_cmd)}")
            
            result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logging.info(f"[DOCKER]  Segmentation complete for {participant_id}")
                return True
            else:
                logging.error(f"[DOCKER]  Docker failed for {participant_id} (exit code: {result.returncode})")
                logging.debug(f"[DOCKER] STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logging.error(f"[DOCKER]  Timeout for {participant_id} (exceeded 1 hour)")
            return False
        except Exception as e:
            logging.error(f"[DOCKER]  Unexpected error for {participant_id}: {e}")
            return False

    def process_participant(self, participant_id: str, metadata_df: pd.DataFrame) -> bool:
        """
        Process a single participant through the complete pipeline.
        
        Pipeline:
          1. Extract frame range from CSV
          2. Validate prompt file exists
          3. Extract frames locally (FFmpeg)
          4. Generate per-participant config
          5. Launch Docker container
          
        Args:
            participant_id: Participant ID (e.g., "510")
            metadata_df: DataFrame with frame ranges
            
        Returns:
            success: True if participant completed successfully
        """
        logging.info(f"\n{'='*70}")
        logging.info(f"[PROCESS] Participant {participant_id}")
        logging.info(f"{'='*70}")
        
        try:
            # Step 1: Get frame range from CSV
            participant_row = metadata_df[metadata_df["Participant_ID"] == int(participant_id)]
            if participant_row.empty:
                logging.warning(f"[SKIP] {participant_id} not found in metadata CSV")
                self.results["skipped"].append(participant_id)
                return False
            
            start_frame = int(participant_row.iloc[0]["Start_Frame"])
            end_frame = int(participant_row.iloc[0]["End_Frame"])
            logging.debug(f"[PROCESS] Frame range: {start_frame}-{end_frame}")
            
            # Step 2: Validate prompt file
            prompt_exists, prompt_file = self._validate_prompt_file(participant_id)
            if not prompt_exists:
                logging.error(f"[FAIL] Prompt file missing for {participant_id}")
                self.results["failed"].append((participant_id, "Prompt file not found"))
                return False
            
            # Step 3: Extract frames
            extract_ok, frames_dir = self._extract_frames(participant_id, start_frame, end_frame)
            if not extract_ok:
                logging.error(f"[FAIL] Frame extraction failed for {participant_id}")
                self.results["failed"].append((participant_id, "Frame extraction failed"))
                return False
            
            # Step 4: Generate per-participant config
            config_ok, config_file = self._generate_participant_config(
                participant_id, start_frame, end_frame, frames_dir
            )
            if not config_ok:
                logging.error(f"[FAIL] Config generation failed for {participant_id}")
                self.results["failed"].append((participant_id, "Config generation failed"))
                return False
            
            # Step 5: Run Docker segmentation
            docker_ok = self._run_docker_segmentation(
                participant_id, frames_dir, prompt_file, config_file
            )
            
            if docker_ok:
                expected_count = self._expected_frame_count(start_frame, end_frame)
                outputs_ok = self._validate_segmentation_outputs(participant_id, expected_count)
                if not outputs_ok:
                    logging.error(f"[FAIL] Segmentation output validation failed for {participant_id}")
                    self.results["failed"].append((participant_id, "Segmentation output validation failed"))
                    return False

                logging.info(f"[SUCCESS]  Participant {participant_id} completed")
                self.results["successful"].append(participant_id)
                return True
            else:
                logging.error(f"[FAIL] Docker segmentation failed for {participant_id}")
                self.results["failed"].append((participant_id, "Docker segmentation failed"))
                return False
                
        except Exception as e:
            logging.error(f"[FAIL] Unexpected error for {participant_id}: {e}", exc_info=True)
            self.results["failed"].append((participant_id, str(e)))
            return False

    def run_batch(self) -> int:
        """
        Execute batch processing across all participants.
        
        Implements fault tolerance: continues processing even if individual 
        participants fail. Provides comprehensive summary at completion.
        
        Returns:
            exit_code: 0 if all successful, 1 if any failed
        """
        try:
            metadata_df = self._load_metadata_csv()
        except Exception as e:
            logging.error(f"[FATAL] Cannot proceed without metadata: {e}")
            return 1
        
        logging.info(f"\n[START] Processing {len(self.participant_ids)} participants...")
        
        for i, pid in enumerate(self.participant_ids, 1):
            try:
                self.process_participant(pid, metadata_df)
            except KeyboardInterrupt:
                logging.warning("\n[INTERRUPT] User interrupted batch processing")
                break
            except Exception as e:
                logging.error(f"[FATAL] Unrecoverable error for {pid}: {e}", exc_info=True)
                self.results["failed"].append((pid, "Unrecoverable error"))
        
        self._print_summary()
        
        # Return exit code: 0 if all successful, 1 if any failed
        return 0 if len(self.results["failed"]) == 0 else 1

    def _print_summary(self) -> None:
        """Print final summary table showing success/failure for all participants."""
        logging.info(f"\n{'='*70}")
        logging.info("[SUMMARY] Batch Processing Complete")
        logging.info(f"{'='*70}")
        
        total = len(self.participant_ids)
        successful = len(self.results["successful"])
        failed = len(self.results["failed"])
        skipped = len(self.results["skipped"])
        
        # Build summary table
        summary_data = []
        
        # Successful participants
        for pid in self.results["successful"]:
            summary_data.append([pid, " SUCCESS", ""])
        
        # Failed participants
        for item in self.results["failed"]:
            pid = item[0] if isinstance(item, tuple) else item
            reason = item[1] if isinstance(item, tuple) and len(item) > 1 else "Unknown error"
            summary_data.append([pid, " FAILED", reason])
        
        # Skipped participants
        for pid in self.results["skipped"]:
            summary_data.append([pid, "SKIPPED", "Not in metadata"])
        
        # Print table
        headers = ["Participant ID", "Status", "Details"]
        table_str = tabulate(summary_data, headers=headers, tablefmt="grid")
        logging.info(f"\n{table_str}")
        
        # Print statistics
        logging.info(f"\n[STATS] Total:      {total}")
        logging.info(f"[STATS] Successful: {successful} ({100*successful//max(total,1)}%)")
        logging.info(f"[STATS] Failed:     {failed}")
        logging.info(f"[STATS] Skipped:    {skipped}")
        
        if failed > 0:
            logging.warning(f"\n[ACTION] {failed} participant(s) failed. Review batch_processing.log for details.")
            logging.warning(f"[ACTION] To re-run failed participants, edit batch_config.yaml and set:")
            logging.warning(f"[ACTION]   mode: \"list\"")
            logging.warning(f"[ACTION]   participant_ids_list: {[item[0] for item in self.results['failed']]}")
        else:
            logging.info(f"\n[COMPLETE]  All {total} participants processed successfully!")
        
        logging.info(f"{'='*70}\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Entry point for batch processor."""
    config_file = sys.argv[1] if len(sys.argv) > 1 else "batch_config.yaml"
    
    try:
        processor = BatchProcessor(config_file)
        exit_code = processor.run_batch()
        sys.exit(exit_code)
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
