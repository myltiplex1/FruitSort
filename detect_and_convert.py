import cv2
import numpy as np
from detection import load_model, infer_on_frame, class_names
from conversion import load_homography, pixel_to_world

def main():
    model = load_model()
    H = load_homography()
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        annotated, det = infer_on_frame(model, frame)
        if len(det):
            for *xyxy, conf, cls in det:
                class_name = class_names[int(cls)]
                center_u = (xyxy[0] + xyxy[2]) / 2
                center_v = (xyxy[1] + xyxy[3]) / 2
                x, y = pixel_to_world(H, center_u, center_v)
                print(f"Detected {class_name} at ({x:.2f}, {y:.2f}) cm, conf={conf:.2f}")
        
        cv2.imshow("Detection", annotated)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()