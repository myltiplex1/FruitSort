# FruitSort

## Autonomous Fruit Sorting System using YOLO and a 4DOF Robotic Arm

This project is an autonomous system that uses computer vision to detect fruits, classify them, and sort them using a 4-degree-of-freedom (4DOF) robotic arm. The detection is performed using a YOLOv5 model, the inverse kinematics for the arm's movements are calculated in Python and the arm is controlled by Arduino.

## Demo video

Watch the project demo on Vimeo: [FruitSort](https://vimeo.com/1116588705?share=copy)

## Features

*   **Fruit Detection:** Utilizes a YOLOv5 model to detect and classify different types of fruits from a webcam feed.
*   **Dataset / Classes:** The model and inference code use three fruit classes: `apple`, `banana`, and `orange`.
*   **Inverse Kinematics:** The system calculates the required joint angles for the robotic arm using a pure Python implementation of Inverse Kinematics (IK) in `main_serial.py`. `main_test.py` provides tests and simulations for IK and the pick-and-place flow.
*   **Robotic Arm Control:** A 4DOF robotic arm, controlled by an Arduino and an Adafruit-style PWM servo shield, physically sorts the detected fruits.
*   **Serial Communication:** A Python script communicates with the Arduino via serial communication to send servo angle commands for the sorting operation.
*   **Calibration:** Includes scripts for camera and robotic arm calibration to ensure accurate mapping from image coordinates to real world coordinates.

## Dataset / Classes

- Number of classes: 3
- Classes: `apple`, `banana`, `orange`

Notes:
- The system currently sorts only these three fruit types. 
- Use `xml_to_coco.py` to convert Pascal VOC/XML annotations to YOLO/COCO-compatible label formats for training.

## Project Structure

Here is an overview of the important files and directories in this project:

| File/Folder | Description |
| --- | --- |
| `main_serial.py` | The main script that orchestrates the entire process: captures video, runs detection, calculates inverse kinematics (IK) to find joint angles,adds the required offsets and sends commands to the Arduino. It also contains the pick-and-place logic, priority order, and drop-point mapping. |
| `serial_arm/serial_arm.ino` | The Arduino sketch that receives commands from `main_serial.py` to control the robotic arm's servos (uses Adafruit PWM servo driver). |
| `manual_servo_test/manual_servo_test.ino` | Arduino sketch used to manually test and find servo offsets for each joint. |
| `detection.py` / `detection1.py` | Detection/inference wrappers for the YOLOv5 model (`detection1.py` is the PyTorch `.pt`-oriented variant). |
| `infer_onnx.py` / `infer_pt.py` | Standalone inference examples/scripts for ONNX and PyTorch models. |
| `main_test.py` | Test harness for IK, drop positions, and serial communication; useful for offline testing before running the live system. |
| `caliberation_save.py` | Script to save calibration points and homography for mapping image to arm coordinates. |
| `caliberation_graph.py` | Script to visualize calibration data and check fit/accuracy. |
| `camera_test.py` | Quick webcam check script. |
| `xml_to_coco.py` | Utility to convert XML annotations into YOLO/COCO-compatible labels. Update the `classes` list inside this file when you change classes. |
| `yolovfruit_cpu.txt` / `yolovfruit_gpu.txt` | Project setup instructions and/or package recommendations for CPU vs GPU environments; they also show dataset/class config snippets used during development. |
| `runs/` | Output from training and detection runs: saved weights (`best.pt`, `best.onnx`), metrics, and visualization images. |

## Hardware Requirements

*   A computer (with a GPU for better performance, but CPU is also supported).
*   Webcam
*   4DOF Robotic Arm with `MG966R` servos (or similar)
*   Arduino (or a compatible microcontroller)
*   Arduino Servo Shield (Adafruit PWM driver compatible)
*   12V 2A Power Adapter
*   LM2596 DC-DC Buck Converter

A detailed schematic of the hardware connections can be found in schematic_diagram.pdf.

### Power Setup (how I powered the servos)

The `MG966R` servos require a stable external supply. The wiring/power approach used in this project:

1.  A **12V 2A adapter** supplies the main power input.
2.  An **LM2596 DC-DC buck converter** steps the 12V down to ~6V (suitable for MG966R servos).
3.  The 6V output from the buck converter powers the **Arduino Servo Shield** (the servo shield then distributes power to the servos).
4.  The Arduino itself is not used as the servo power source — the servo shield gets its supply from the buck converter to avoid drawing excessive current from the Arduino board.

This arrangement prevents servo brownouts and keeps the Arduino stable. Refer to `schematic_diagram.pdf' for the full wiring diagram and connector pinouts.

## Software and Installation

1.  **Clone the repository:**
    ```powershell
    git clone https://github.com/myltiplex1/FruitSort.git
    cd FruitSort
    ```

2.  **Set up Arduino:**
    *   Open `serial_arm/serial_arm.ino` (or `manual_servo_test/manual_servo_test.ino`) in the Arduino IDE.
    *   Install required libraries (e.g., `Adafruit_PWMServoDriver`).
    *   Upload the sketch to the Arduino.

3.  **Install Python dependencies and Train YOLO Model:**
    *   If you are using a **GPU** follow the steps suggested in `yolovfruit_gpu.txt`
    *   If you are using a **CPU** follow the steps suggested in `yolovfruit_cpu.txt`

## Usage

1.  **Calibrate the System:**
    *   Run the manual servo test sketch to find per-servo offsets (`manual_servo_test/manual_servo_test.ino`).
    *   Collect camera to world calibration points and save them using `caliberation_save.py`. Visualize/verify with `caliberation_graph.py`.
    *   Confirm webcam works with `camera_test.py`.

2.  **Run detection / inference only (optional):**
    *   Use `infer_onnx.py` or `infer_pt.py` to run a single-image inference and confirm the model predicts the three classes (`apple`, `banana`, `orange`).

3. **Test detection and conversion:**
    *   Use `detect_and_convert.py` to test the predictions and confirm the real world coordinates are correct.

4.  **Run the full system:**
    *   Connect the Arduino with the servo shield (powered from the buck converter).
    *   Run:
    ```powershell
    python main_serial.py
    ```
    *   Observe detection boxes, the system will compute real world coordinates, solve IK in Python, add offsets and send servo commands to the Arduino to pick, lift and place fruits.

## Training the YOLOv5 Model

The YOLOv5 model was trained on a custom dataset containing the three fruit classes described above. The `runs` directory contains artifacts from the training process (weights: `best.pt`, `best.onnx`, metrics and visualization images).

To retrain:
1. Prepare data in Pascal VOC/XML or YOLO txt format.
2. Use `xml_to_coco.py` to convert XML labels to YOLO-style text labels if needed (edit the `classes` list inside the script to match your categories).
3. Train using your prepared dataset and config (adjust `nc` and `names` accordingly) by following the steps in `yolovfruit_gpu.txt` if you are using a **GPU** or `yolovfruit_cpu.txt` if you are using a **CPU**.

## Troubleshooting & Tips

*  If servos jitter or the Arduino resets under load, ensure servo power is from the LM2596-buck -> servo shield and not the Arduino 5V rail.
*  Test the IK with `main_test.py` before connecting to hardware.
*  If detection is slow on CPU, use the ONNX path or smaller models, or run on a machine with a CUDA-capable GPU and the packages from `yolovfruit_gpu.txt`.

## Where to look next (quick links)

*  `main_serial.py` — full system orchestration, IK and serial comms.
*  `serial_arm/serial_arm.ino` — Arduino servo control sketch.
*  `manual_servo_test/manual_servo_test.ino` — servo offset tuning.
*  `detection.py`, `detection1.py`, `infer_onnx.py`, `infer_pt.py` — detection & inference examples.
*  `xml_to_coco.py` — annotation conversion; check `classes` array here when changing dataset classes.

---
