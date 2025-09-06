# FruitSort
## Autonomous Fruit Sorting System using YOLO and a 4DOF Robotic Arm

This project is an autonomous system that uses computer vision to detect fruits, classify them, and sort them using a 4-degree-of-freedom (4DOF) robotic arm. The detection is performed using a YOLOv5 model, the inverse kinematics for the arm's movements are calculated in Python, and the arm is controlled by an Arduino.

## Features

*   **Fruit Detection:** Utilizes a YOLOv5 model to detect and classify different types of fruits from a webcam feed.
*   **Inverse Kinematics:** The system calculates the required joint angles for the robotic arm using a pure Python implementation of Inverse Kinematics (IK).
*   **Robotic Arm Control:** A 4DOF robotic arm, controlled by an Arduino and a servo shield, physically sorts the detected fruits.
*   **Serial Communication:** A Python script communicates with the Arduino via serial communication to send commands for the sorting operation.
*   **Calibration:** Includes scripts for camera and robotic arm calibration to ensure accurate picking and placing.

## Project Structure

Here is an overview of the important files and directories in this project:

| File/Folder | Description |
| --- | --- |
| `main_serial.py` | The main script that orchestrates the entire process: captures video, runs detection, calculates inverse kinematics to find joint angles, and sends commands to the Arduino. |
| `serial_arm/serial_arm.ino` | The primary Arduino sketch that receives commands from `main_serial.py` to control the robotic arm's servos. |
| `detection.py` / `detection1.py` | Scripts for testing the fruit detection functionality of the YOLOv5 model (ONNX and PyTorch versions). |
| `infer_onnx.py` / `infer_pt.py` | Scripts for running inference with the trained YOLOv5 models (`.onnx` and `.pt` formats). |
| `main_test.py` | A test script for verifying the inverse kinematics (IK) calculations and the serial communication with the Arduino. |
| `manual_servo_test/` | Contains an Arduino sketch (`manual_servo_test.ino`) for manually testing and finding the offset angles for each servo of the robotic arm. |
| `caliberation_save.py` | Script to save calibration data. |
| `caliberation_graph.py` | Script to visualize calibration data. |
| `camera_test.py` | A simple script to test the webcam functionality. |
| `xml_to_coco.py` | A utility script to convert annotations from XML format to COCO format, which is useful for training the YOLO model. |
| `yolovfruit_cpu.txt` | A `requirements.txt` file with the necessary Python packages for running the project on a CPU. |
| `yolovfruit_gpu.txt` | A `requirements.txt` file with the necessary Python packages for running the project on a machine with a GPU. |
| `runs/` | This directory contains the output from training and detection runs, including saved models, logs, and sample images. |

## Hardware Requirements

*   A computer (with a GPU for better performance, but CPU is also supported).
*   Webcam
*   4DOF Robotic Arm with `MG966R` servos (or similar)
*   Arduino (or a compatible microcontroller)
*   Arduino Servo Shield
*   12V 2A Power Adapter
*   LM2596 DC-DC Buck Converter

A detailed schematic of the hardware connections can be found in `schematic.pdf`.

### Power Setup

The `MG966R` servos require a stable power supply. The system is powered as follows:
1.  A **12V 2A adapter** provides the main power.
2.  This is connected to an **LM2596 DC-DC buck converter** to step the voltage down to 6V.
3.  The 6V output from the buck converter is used to power the **Arduino Servo Shield**, which in turn powers the servos. This ensures the servos have enough current without drawing too much from the Arduino's onboard regulator.

## Software and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd FruitSort
    ```

2.  **Set up Arduino:**
    *   Open `serial_arm/serial_arm.ino` in the Arduino IDE.
    *   Install any required libraries for the servos.
    *   Upload the sketch to your Arduino.

3.  **Install Python dependencies:**
    *   If you are using a **GPU**, install the packages from `yolovfruit_gpu.txt`:
        ```bash
        pip install -r yolovfruit_gpu.txt
        ```
    *   If you are using a **CPU**, install the packages from `yolovfruit_cpu.txt`:
        ```bash
        pip install -r yolovfruit_cpu.txt
        ```

## Usage

1.  **Calibrate the System:**
    *   Run `manual_servo_test/manual_servo_test.ino` to determine the servo offsets.
    *   Run `caliberation_save.py` and `camera_test.py` to perform camera and spatial calibration.

2.  **Run the Main Application:**
    *   Make sure the Arduino with `serial_arm.ino` is connected to your computer.
    *   Execute the main script:
        ```bash
        python main_serial.py
        ```

## Training the YOLOv5 Model

The YOLOv5 model was trained on a custom dataset of fruits. The `runs/train/` directory contains artifacts from the training process, such as weights (`best.pt`, `best.onnx`), metrics, and learning curves.

If you want to retrain the model on your own dataset, you can use the `xml_to_coco.py` script to get your data into the right format and then follow the standard YOLOv5 training procedures.
