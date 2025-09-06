import cv2
import numpy as np
import math
import keyboard
import time
import serial
from detection import load_model, infer_on_frame, class_names
from conversion import load_homography, pixel_to_world

# Config
base_x, base_y = 0, -11.8  # Base is 11.8 cm behind homography origin
l1, l2, l3 = 0, 12, 16  # Arm link lengths in cm
z_ground = 0.0  # Target z at table level
lift_height = 3.0  # cm above ground for lifting
z_lift = z_ground + lift_height
drop_positions = {
    "apple": (15, 0),   # Drop position for apple (base angle 0°)
    "orange": (15, 0),  # Drop position for orange (same as apple, base angle 0°)
    "banana": (-15, 0)  # Drop position for banana (base angle 180°)
}

# Global flag to track pick sequence
is_pick_sequence = False

# Serial port configuration
SERIAL_PORT = 'COM3'  # Adjust to your Arduino's serial port
BAUD_RATE = 9600

def inverse_kinematics(l1, l2, l3, x, y, z, elbow_up=True):
    """
    Compute inverse kinematics for a 3DOF arm.
    Inputs: l1, l2, l3 (link lengths in cm), x, y, z (target in cm), elbow_up (configuration).
    Returns: (theta1, theta2, theta3) in radians or None if unreachable.
    """
    r = math.sqrt(x**2 + y**2)
    if r < 0.01:  # Avoid singularity
        print("Singularity at z-axis (r too small)")
        return None
    theta1 = math.atan2(y, x)
    
    s = math.sqrt(r**2 + (z - l1)**2)
    if s > l2 + l3 or s < abs(l2 - l3):
        print(f"Target ({x:.2f}, {y:.2f}, {z:.2f}) cm is unreachable (s={s:.2f} cm)")
        return None
    
    cos_theta3 = (l2**2 + l3**2 - s**2) / (2 * l2 * l3)
    if abs(cos_theta3) > 1:
        print("Invalid theta3 (cos_theta3 out of range)")
        return None
    sin_theta3 = -math.sqrt(1 - cos_theta3**2) if elbow_up else math.sqrt(1 - cos_theta3**2)
    theta3 = math.atan2(sin_theta3, cos_theta3)
    
    theta2 = math.atan2(z - l1, r) - math.atan2(l3 * sin_theta3, l2 + l3 * cos_theta3)
    
    return theta1, theta2, theta3

def apply_joint_offsets(angles_rad):
    """
    Convert IK angles to robot servo angles with specified limits and orientations.
    Inputs: angles_rad (theta1, theta2, theta3) in radians.
    Returns: (base, shoulder, elbow) in degrees.
    """
    angles_deg = [math.degrees(angle) for angle in angles_rad]
    print(f"Raw IK angles (deg): theta1={angles_deg[0]:.2f}, theta2={angles_deg[1]:.2f}, theta3={angles_deg[2]:.2f}")
    
    base = angles_deg[0]
    if base < 0:
        base += 360
    base = np.clip(base, 0, 180)
    
    shoulder = -angles_deg[1] + 160
    shoulder = np.clip(shoulder, 80, 160)
    
    elbow = -angles_deg[2]/9 + 60
    elbow = np.clip(elbow, 60, 120)
    
    print(f"Mapped angles (deg): base={base:.2f}, shoulder={shoulder:.2f}, elbow={elbow:.2f}")
    return base, shoulder, elbow

def print_command(joint, angle, ser=None):
    """
    Send command to robot via serial and print for debugging, applying offsets for pick sequence.
    Inputs: joint (str), angle (float), ser (serial.Serial object or None).
    """
    global is_pick_sequence
    adjusted_angle = angle
    
    # Apply offsets only during pick sequence
    if is_pick_sequence:
        if joint == "BASE":
            if angle > 90:
                adjusted_angle = angle + 5
            else:
                adjusted_angle = angle - 15
        elif joint == "SHOULDER":
            adjusted_angle = angle + 85
        elif joint == "ELBOW":
            adjusted_angle = angle - 55
    
    # Clip angles to joint limits
    if joint == "BASE":
        adjusted_angle = np.clip(adjusted_angle, 0, 180)
    elif joint == "SHOULDER":
        adjusted_angle = np.clip(adjusted_angle, 70, 180)
    elif joint == "ELBOW":
        adjusted_angle = np.clip(adjusted_angle, 0, 120)
    elif joint == "GRIP":
        adjusted_angle = np.clip(adjusted_angle, 0, 90)
    
    # Print for debugging
    print(f"Would send: {joint}:{int(adjusted_angle)}")
    
    # Send to Arduino if serial connection is available
    if ser is not None and ser.is_open:
        try:
            ser.write(f"{joint}:{int(adjusted_angle)}\n".encode())
            ser.flush()  # Ensure the command is sent
            time.sleep(0.1)  # Small delay to allow Arduino to process
        except serial.SerialException as e:
            print(f"Serial error: {e}")

def move_to_home(ser=None):
    """
    Move the arm to home position.
    """
    print("Moving to home position (BASE:90, SHOULDER:70, ELBOW:0, GRIP:0)")
    global is_pick_sequence
    is_pick_sequence = False
    print_command("BASE", 90, ser)
    print_command("SHOULDER", 70, ser)
    print_command("ELBOW", 0, ser)
    time.sleep(2)  # Wait for arm to stabilize
    print_command("GRIP", 0, ser)
    time.sleep(1)  # Wait for gripper to open

def pick_and_place_fruit(class_name, center_u, center_v, H, ser=None):
    """
    Perform pick-and-place for a detected fruit.
    Inputs: class_name (str), center_u, center_v (pixel coordinates), H (homography matrix), ser (serial.Serial object or None).
    Returns: True if successful, False if unreachable.
    """
    global is_pick_sequence
    real_x, real_y = pixel_to_world(H, center_u, center_v)
    print(f"Detected {class_name} at ({real_x:.2f}, {real_y:.2f}) cm")

    ik_x = real_x - base_x
    ik_y = real_y - base_y
    print(f"Adjusted position for IK: ({ik_x:.2f}, {ik_y:.2f}, {z_ground:.2f}) cm")

    # Pick: Move to z_ground, close gripper
    is_pick_sequence = True
    # Check if Y-axis distance exceeds 13 cm
    if abs(ik_y) > 24:
        print(f"Y-axis distance ({abs(ik_y):.2f} cm) exceeds 13 cm, using fixed angles for picking")
        angles = inverse_kinematics(l1, l2, l3, ik_x, ik_y, z_ground)
        if angles:
            base, _, _ = apply_joint_offsets(angles)
            print_command("BASE", base, ser)
            print_command("SHOULDER", 175, ser)
            print_command("ELBOW", 0, ser)
            time.sleep(2)  # Wait for arm to stabilize
            print_command("GRIP", 0, ser)
            time.sleep(1)  # Wait for gripper to close
            print_command("GRIP", 90, ser)
        else:
            print("Cannot reach pick position")
            return False
    else:
        angles = inverse_kinematics(l1, l2, l3, ik_x, ik_y, z_ground)
        if angles:
            base, shoulder, elbow = apply_joint_offsets(angles)
            print_command("BASE", base, ser)
            print_command("SHOULDER", shoulder, ser)
            print_command("ELBOW", elbow, ser)
            time.sleep(2)  # Wait for arm to stabilize
            print_command("GRIP", 0, ser)
            time.sleep(1)  # Wait for gripper to close
            print_command("GRIP", 90, ser)
        else:
            print("Cannot reach pick position")
            return False

    # Lift: Move to z_lift
    is_pick_sequence = False
    angles = inverse_kinematics(l1, l2, l3, ik_x, ik_y, z_lift)
    if angles:
        base, shoulder, elbow = apply_joint_offsets(angles)
        print_command("BASE", base, ser)
        print_command("SHOULDER", shoulder - 10, ser)
        print_command("ELBOW", elbow - 0, ser)
        time.sleep(2)  # Wait for arm to stabilize
    else:
        print("Cannot reach lift position")
        return False

    # Drop: Move to drop position at z_ground
    drop_x, drop_y = drop_positions[class_name]
    print(f"{class_name.capitalize()} detected! Moving to drop position ({drop_x}, {drop_y}, {z_ground})")
    angles = inverse_kinematics(l1, l2, l3, drop_x, drop_y, z_ground)
    if angles:
        base, shoulder, elbow = apply_joint_offsets(angles)
        print_command("BASE", base, ser)
        print_command("SHOULDER", shoulder, ser)
        print_command("ELBOW", elbow, ser)
        time.sleep(2)  # Wait for arm to stabilize
        print_command("GRIP", 0, ser)
        time.sleep(1)  # Wait for gripper to open
    else:
        print("Cannot reach drop position")
        return False

    time.sleep(1)  # Wait after drop
    return True

def main():
    model = load_model()
    H = load_homography()
    
    # Initialize serial connection
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to Arduino on {SERIAL_PORT}")
        time.sleep(2)  # Allow Arduino to initialize
    except serial.SerialException as e:
        print(f"Failed to connect to Arduino on {SERIAL_PORT}: {e}")
        print("Continuing in simulation mode (printing commands only).")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        if ser is not None and ser.is_open:
            ser.close()
        return

    print("Webcam opened. Press 'p' to start autonomous sorting, 'h' to return to home position, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        annotated, det = infer_on_frame(model, frame)
        print(f"Detections: {len(det)}")
        cv2.imshow("Detection", annotated)

        if keyboard.is_pressed('q'):
            print("Quitting...")
            break
        elif keyboard.is_pressed('h'):
            move_to_home(ser)
        elif keyboard.is_pressed('p'):
            print("Starting autonomous sorting in 3 seconds...")
            time.sleep(3)
            print("Sorting sequence started.")
            
            # Priority order: orange, banana, apple
            fruit_priority = ["orange", "banana", "apple"]
            for fruit in fruit_priority:
                # Check for the current fruit in detections
                for *xyxy, conf, cls in det:
                    class_name = class_names[int(cls)].lower()
                    if class_name == fruit:
                        center_u = (xyxy[0] + xyxy[2]) / 2
                        center_v = (xyxy[1] + xyxy[3]) / 2
                        if pick_and_place_fruit(class_name, center_u, center_v, H, ser):
                            # Return to home after successful pick-and-place
                            move_to_home(ser)
                            # Capture a new frame to update detections
                            ret, frame = cap.read()
                            if not ret:
                                print("Failed to capture frame")
                                break
                            annotated, det = infer_on_frame(model, frame)
                            print(f"Detections after picking {class_name}: {len(det)}")
                            cv2.imshow("Detection", annotated)
                            break  # Move to next fruit type
                else:
                    print(f"No {fruit} detected, skipping to next fruit.")

            # After sorting all fruits, return to home
            print("Sorting sequence completed.")
            move_to_home(ser)

        cv2.waitKey(10)

    cap.release()
    cv2.destroyAllWindows()
    if ser is not None and ser.is_open:
        ser.close()
        print("Serial connection closed.")

if __name__ == "__main__":
    main()