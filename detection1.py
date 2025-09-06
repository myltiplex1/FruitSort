import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np

# Setup paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import YOLOv5 utilities
from models.common import DetectMultiBackend
from utils.dataloaders import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from ultralytics.utils.plotting import Annotator, colors

# Class labels
class_names = ["apple", "banana", "orange"]

def load_model(
    weights="runs/train/exp2/weights/best.pt",
    device='cpu'
):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    return model

def infer_on_frame(
    model,
    im0,
    imgsz=(640, 640),
    conf_thres=0.6,
    iou_thres=0.4
):
    stride, names, pt = model.stride, class_names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Preprocess
    im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(model.device)
    im = im.float() / 255.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)

    # Inference
    pred = model(im)
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    det = pred[0] if pred else torch.tensor([])

    annotator = Annotator(im0.copy(), line_width=3, example=str(names))
    if len(det):
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in det:
            label = f"{names[int(cls)]} {conf:.2f}"
            annotator.box_label(xyxy, label, color=colors(int(cls), True))

    annotated_im0 = annotator.result()

    return annotated_im0, det

if __name__ == "__main__":
    model = load_model()
    model.warmup(imgsz=(1, 3, 640, 640))  # Warm up model
    cap = cv2.VideoCapture(1)  # Use webcam (index 1)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        annotated, det = infer_on_frame(model, frame)
        if len(det):
            for *xyxy, conf, cls in det:
                print(f"Detected {class_names[int(cls)]}: xyxy={xyxy}, conf={conf:.2f}")
        cv2.imshow("Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()