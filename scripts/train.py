#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", default="yolo11m.pt", help="ex: yolo11m.pt / yolo26m.pt")
    p.add_argument("--data", type=Path, default=ROOT / "config" / "data.yaml")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="based on VRAM")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None, help="ex: 0, 0,1 or cpu (default: let Ultralytics choose)")
    p.add_argument("--cache", choices=["disk", "ram", "none"], default="disk")
    p.add_argument("--close-mosaic", type=int, default=5)
    p.add_argument("--save-period", type=int, default=5)
    p.add_argument("--project", type=Path, default=ROOT / "runs")
    p.add_argument("--name", default=None, help="default: wildlive_<model name>")
    p.add_argument("--custom-aug", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    name = args.name or f"wildlive_{Path(args.model).stem}"
    last = args.project / name / "weights" / "last.pt"

    if args.resume:
        if not last.exists():
            sys.exit(f"not found {last} .")
        YOLO(str(last)).train(resume=True)
        return

    if not args.data.exists():
        sys.exit(f"not found {args.data}. Run scripts/convert_into_yoloformat.py first.")

    kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        cache=False if args.cache == "none" else args.cache,
        close_mosaic=args.close_mosaic,
        save=True,
        save_period=args.save_period,
        project=str(args.project),
        name=name,
        exist_ok=True,
        plots=True,
    )
    if args.device is not None:
        kwargs["device"] = args.device

    if args.custom_aug:
        sys.path.insert(0, str(ROOT / "scripts"))
        from augmentation import build_custom_aug
        kwargs["augmentations"] = build_custom_aug()

    model = YOLO(args.model)
    model.train(**kwargs)


if __name__ == "__main__":
    main()