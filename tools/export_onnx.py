from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument("--output", required=True, help="Destination .onnx path")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--int8", action="store_true", help="Quantize to INT8")
    parser.add_argument("--half", action="store_true", help="Quantize to FP16")
    parser.add_argument("--data", default=None, help="data.yaml for INT8 calibration")
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    export_kwargs = dict(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,
    )
    if args.int8:
        export_kwargs["int8"] = True
        if args.data:
            export_kwargs["data"] = args.data
    elif args.half:
        export_kwargs["half"] = True

    out_path = Path(model.export(**export_kwargs))
    print(f"Exported to: {out_path}")

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path, dst)
    print(f"Copied to:   {dst}")


if __name__ == "__main__":
    main()

