import os
import sys
from pathlib import Path

import torch
import cv2
import numpy as np

# === Setup ===
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import (
    cv2, non_max_suppression, scale_boxes, check_img_size,
    increment_path, xyxy2xywh
)
from utils.torch_utils import select_device
from ultralytics.utils.plotting import Annotator, colors

# === Class labels ===
class_names = ["apple", "banana", "orange"]

def run_inference(
    weights="runs/train/exp2/weights/optimizedonnx/best.onnx",
    source="dataset/images/val/212.jpg",
    imgsz=(640, 640),
    conf_thres=0.7,
    iou_thres=0.45,
    save=False  # 👈 Add this flag
):
    save_dir = increment_path(Path("runs/infer"), exist_ok=True) if save else None
    if save:
        os.makedirs(save_dir, exist_ok=True)

    is_webcam = source.isnumeric()
    device = select_device('cpu')
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, class_names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt) if is_webcam else LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        # Ensure size is correct
        if im.shape[2:] != torch.Size([640, 640]):
            im = torch.nn.functional.interpolate(im, size=(640, 640), mode='bilinear', align_corners=False)

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):
            im0 = im0s[i].copy() if is_webcam else im0s
            annotator = Annotator(im0, line_width=3, example=str(names))
            if det is not None and len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    label = f"{names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))
            im0 = annotator.result()

            # === Only save if requested ===
            if not is_webcam:
                if save:
                    save_path = str(save_dir / Path(path).name)
                    cv2.imwrite(save_path, im0)
                    print(f"Saved to {save_path}")
                cv2.imshow("Result", im0)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                cv2.imshow("Webcam", im0)
                if cv2.waitKey(1) == ord("q"):
                    break

    if is_webcam:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # run_inference()
    run_inference(source='0')
