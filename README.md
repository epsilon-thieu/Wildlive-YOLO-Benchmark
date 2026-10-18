# Wildlive-YOLO-Benchmark

## Introduction

This repository provides the training and benchmarking pipeline for fine-tuning [Ultralytics](https://github.com/ultralytics/ultralytics)YOLO11 and YOLO26 detectors on the [wildlive dataset](https://dat-nguyenvn.github.io/WildLive/) — drone footage of three wildlife classes (zebra, giraffe, elephant).
This repo covers:

- **Dataset reformatting** — converting the
  [WildLive dataset](https://dat-nguyenvn.github.io/WildLive/) LabelMe
  annotations into the YOLO format, with a fixed video-level
  train/val/test split.
- **Detector fine-tuning** — fine-tuning eight pretrained checkpoints
  (YOLO11 m/l/x and YOLO26 n/s/m/l/x) with a single training script,
  optionally with an extra Albumentations-based augmentation pipeline.
- **Benchmarking** — evaluating every model on held-out test videos and
  reporting precision, recall, mAP and FPS for direct comparison.

#image
> **Note:** The image shows predictions of a fine-tuned model on a test-video frame.

## Install

The source code is implemented as plain Python scripts and runs locally
(a CUDA GPU is recommended). Before training, make sure the WildLive dataset
is prepared as described in the [Data Preparation](#data-preparation) section.

```bash
git clone https://github.com/epsilon-thieu/Wildlive-YOLO-Benchmark.git
cd Wildlive-YOLO-Benchmark
```

Install [PyTorch](https://pytorch.org) for your CUDA version first, then:

```bash
pip install -U ultralytics "albumentations>=2.0"
```

### Detect

The pretrained checkpoints (e.g. `yolo11m.pt`, `yolo26m.pt`) are downloaded
automatically by Ultralytics the first time they are requested.

#### Training

Run:
```bash
python scripts/train.py --model yolo11m.pt
python scripts/train.py --model yolo26m.pt --custom-aug
```

Models trained in this repo: `yolo11m`, `yolo11l`, `yolo11x`, `yolo26n`,
`yolo26s`, `yolo26m`, `yolo26l`, `yolo26x`. Each run is saved to
`runs/wildlive_<model>/`.

`--custom-aug` adds the Albumentations pipeline defined in
`scripts/augmentation.py` (noise, blur, color, weather and light geometric
transforms, each applied with a probability of 5-8%).

> **Note:** The reported results were trained with `--custom-aug`:.

#### Optional
This section is for resuming training if you want to continue from where you left off in a previous training run.

Run the same command with `--resume`:
```bash
python scripts/train.py --model yolo26n.pt --resume
```
It continues from `runs/wildlive_yolo26n/weights/last.pt`. Only `last.pt` is
needed.

## Data Preparation
### Original [WildLive dataset](https://dat-nguyenvn.github.io/WildLive/) structure
```
        wildlive_dataset/
        |---sequence_name1/
                |---frames/
                |       |---frame_0.jpg
                |       |---frame_1.jpg ...
                |---boxid/
                        |---frame_0.json    # LabelMe annotation
                        |---frame_1.json ...
```

### Structure required by Ultralytics YOLO
```
        data/yolo/
        |---images/
        |       |---train/<video>/<video>_frame_000001.jpg
        |       |---val/<video>/...
        |       |---test/<video>/...
        |---labels/
                |---train/<video>/<video>_frame_000001.txt
                |---val/<video>/...
                |---test/<video>/...
```
Each label file contains one line per box: `class x_center y_center width height`
(normalized to [0, 1]). Class IDs: `0 = zebra`, `1 = giraffe`, `2 = elephant`.

### Why this repo exists

Since the original WildLive layout (LabelMe polygons/boxes per frame) differs
from what Ultralytics expects, this repo integrates a conversion script that
automatically transforms the original structure into the YOLO format.

To use the script, download the dataset and place the video folders in the
following layout:
```
        Wildlive-YOLO-Benchmark/
        └── data/
                └── raw/
                        ├── DJI_0601_video4/
                        │       ├── frames/
                        │       └── boxid/
                        ├── DJI_0117_video4/
                        └── ...
```
Then run:
```bash
python scripts/convert_into_yoloformat.py
```

### Dataset Split

The split is done **by video**, the test and validation videos are fixed in `scripts/convert_into_yoloformat.py`; all remaining videos are used for training.

| Split | Videos | Frames | Ratio (by frames) |
|---|---|---|---|
| Train | 17 | 15,156 | 79.1% |
| Val   | 3  | 1,891  | 10%  |
| Test  | 2  | 2,096  | 10.9% |
| Total | 22 | 19,143 | 100%  |

- **Val:** `DJI_20230719145427_0002_V_video5`, `DJI_0117_video4`, `DJI_0204_video2`
- **Test:** `DJI_20230719145816_0003_V_video2`, `DJI_0601_video4`

## Result

### Training Results

The plots below show the best value reached by each model.

![Best validation mAP50-95](results\Best mAP50-95.jpg)

| Model | Epochs | Best epoch | Precision (%) | Recall (%) | mAP50 (%) | mAP50-95 (%) |
|---|---|---|---|---|---|---|---|
| YOLO11m | 20 | 16 | 99.6 | 87.1 | 87.0 | 68.1 |
| YOLO11l | 20 | 17 | 99.7 | 87.1 | 87.0 | 67.9 |
| YOLO11x | 8 | 8 | 97.4 | 86.4 | 87.3 | 66.3 |
| YOLO26n | 30 | 9 | 97.9 | 86.4 | 86.9 | 59.8 |
| YOLO26s | 30 | 23 | 99.7 | 87.0 | 87.0 | 67.2 |
| YOLO26m | 20 | 99.9 | 87.1 | 87.0 | 68.7 |
| YOLO26l | 20 | 99.8 | 87.1 | 87.0 | 69.1 |
| YOLO26x | 20 | 15 | 99.8 | 87.0 | 87.0 | 69.0 |

## Citation

If you find this repository helpful for your research, please consider citing our work, as well as the original works it builds upon:

```bibtex
@misc{epsilon2026wildlifeyolobenchmark,
  title  = {Wildlive-YOLO-Benchmark: Fine-tuning and Benchmarking YOLO11 and YOLO26 for Drone-based Wildlife Detection},
  author = {Thieu Le Anh Quy and Nguyen Ngoc Dat},
  year   = {2026},
  url    = {https://github.com/epsilon-thieu/Wildlive-YOLO-Benchmark}
}

@inproceedings{dat2025wildlive,
  title     = {WildLive: Near Real-time Visual Wildlife Tracking onboard {UAV}s},
  author    = {Nguyen Ngoc Dat and Thomas Stuart Richardson and Matthew Watson and Kilian Meier and Jenna Marie Kline and Sid Reid and Guy Maalouf and Duncan Hine and Majid Mirmehdi and Tilo Burghardt},
  booktitle = {5th Workshop on CV4Animals: Computer Vision for Animal Behavior Tracking and Modeling, In conjunction with Computer Vision and Pattern Recognition 2025},
  year      = {2025},
  url       = {https://drive.google.com/file/d/1F5pAptwMLmd78ZhzqTsXVDpK_v-hEeLC/view}
}
```