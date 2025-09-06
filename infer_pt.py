import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes, increment_path, LOGGER
from utils.torch_utils import select_device

def run_inference(
    weights='runs/train/exp2/weights/best.pt',
    source='dataset/images/val/212.jpg',
    imgsz=(640, 640),
    conf_thres=0.7,
    iou_thres=0.45,
    max_det=1000,
    view_img=True,
    save_img=False,
    save_txt=False,
    project='runs/detect',
    name='exp',
    exist_ok=False,
    line_thickness=3,
    classes=["apple", "banana", "orange"]
):
    # Initialize
    device = select_device('cpu')
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    try:
        model = DetectMultiBackend(weights, device=device, dnn=False, fp16=False)
        model.eval()  # Set model to evaluation mode
    except Exception as e:
        LOGGER.error(f"Failed to load model: {e}")
        return
    stride, names = model.stride, classes  # Use provided class names
    imgsz = check_img_size(imgsz, s=stride)

    # Determine source type
    source = str(source)
    webcam = source.isnumeric() or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Load data
    try:
        if webcam:
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=True)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)
    except Exception as e:
        LOGGER.error(f"Failed to load source: {e}")
        return

    # Warmup
    model.warmup(imgsz=(1, 3, *imgsz))

    # Inference
    for path, im, im0s, vid_cap, s in dataset:
        # Preprocess
        im = torch.from_numpy(im).to(device).float() / 255.0
        if len(im.shape) == 3:
            im = im[None]  # Add batch dimension

        # Ensure correct input size
        if im.shape[2:] != torch.Size([640, 640]):
            im = torch.nn.functional.interpolate(im, size=(640, 640), mode='bilinear', align_corners=False)

        # Inference
        try:
            pred = model(im)
            pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
        except Exception as e:
            LOGGER.error(f"Inference failed: {e}")
            continue

        # Process detections
        p = Path(path[0] if webcam else path)
        im0 = im0s[0].copy() if webcam else im0s.copy()
        save_path = str(save_dir / p.name)
        txt_path = str(save_dir / "labels" / p.stem)
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        for det in pred:
            if len(det):
                # Rescale boxes
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Draw boxes
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f"{names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    # Save txt
                    if save_txt:
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

        # Save or show results
        im0 = annotator.result()
        if save_img and not webcam:
            cv2.imwrite(save_path, im0)
        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1 if webcam else 0)

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}processed")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # run_inference()
    run_inference(source="0")