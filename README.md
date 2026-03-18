# METProcessing
Mobile Eye-Tracking Processing with SAM 2

## Overview
The pipeline is divided into these steps:
1. **Extract frames**
2. **Create prompt**
3. **Segment**
4. **Map gaze data**

## Setup
All parameters can be configured inside the `config.yaml` file.

## Docker
```bash
docker build -t sam2 .
```
```bash
docker run --gpus all --rm -v "C:\Users\avery\Desktop\sample data:/data" -v "C:\Users\avery\Desktop\docker_sam2\config.yaml:/config/config.yaml" sam2
```
The `docker run` command starts the segmentation pipeline. Since the container is isolated from your machine, any files required while running need to be mounted beforehand using the `-v` flag, which maps a path on your machine to a path usable inside the container.


This pipeline requies two mounts:
- **Data folder**: containing the frames folder and `gaze_positions.csv`. This will be mapped to `/data` in the container
- **Config file**: the `config.yaml` file which will be mapped to `/config/config.yaml` inside the container.


Inside the command, the left side of the string (before the colon) should be changed to match the path of your data and config files.