# METProcessing
Mobile Eye-Tracking Processing with SAM 2

## Overview
The pipeline is divided into these steps:
1. **Extract frames**
2. **Create prompt**
3. **Segment**
4. **Map gaze data**

## Setup
All parameters for the pipeline can be configured inside the `config.yaml` file or `batch_config.yaml` file, depending on whether the user is processing a few participant IDs manually or wants to process large-scale participant data automatically.

## Docker
### Option 1: Single Participant (Manual)
For processing one participant at a time with explicit Docker commands:

```bash
docker build -t sam2 .
```
```bash
docker run --gpus all --rm -v "C:\Users\avery\Desktop\sample data:/data" -v "C:\Users\avery\Desktop\docker_sam2\config.yaml:/config/config.yaml" sam2
```

The `docker run` command starts the segmentation pipeline. Since the container is isolated from your machine, any files required while running need to be mounted beforehand using the `-v` flag, which maps a path on your machine to a path usable inside the container.

This pipeline requires two mounts:
- **Data folder**: containing the frames folder and `gaze_positions.csv`. This will be mapped to `/data` in the container
- **Config file**: the `config.yaml` file which will be mapped to `/config/config.yaml` inside the container.

Inside the command, the left side of the string (before the colon) should be changed to match the path of your data and config files.

---

### Option 2: Batch Processing
For processing multiple participant IDs in batches, including overnight unattended processing for large datasets:

1. **Configure the batch processor:**
   - Edit `batch_config.yaml` to specify:
     - Participant ID range (e.g., [510, 600]) or explicit list (e.g., [510, 514, 520])
     - Local and Docker data paths
     - Video source paths for FFmpeg extraction (must be named according to participant ID to use pattern-matching, e.g., `participant_510.mp4`)
     - Frame extraction and segmentation settings
     - GPU allocation and logging preferences

2. **Build the Docker image** (if not already built):
   ```bash
   docker build -t sam2 .
   ```

3. **Run the batch processor:**
   ```bash
   python batch_processor.py
   ```

4. **Monitor results:**
   - Console output shows real-time progress
   - Final summary table displays success/failure status for each participant
   - Full execution log saved to `batch_processing.log`

**Key features of batch processing:**
- **Fault tolerance**: If one participant fails, processing automatically continues to the next IDs instead of halting the entire batch
- **Memory safety**: Docker containers reset between participants, preventing Out of Memory (OOM) errors from affecting the runs 
- **Frame extraction**: Automated FFmpeg-based frame extraction from source videos to save time and reduce human error in manual frame handling
- **Dynamic configuration**: Per-participant configs generated automatically at runtime
- **Comprehensive reporting**: ASCII summary table showing which participants succeeded/failed with error details
- **Resumable workflows**: Failed participants can be re-run by editing batch_config.yaml