import cv2
import numpy as np
import math
import keyboard
import time
from detection import load_model, infer_on_frame, class_names
from conversion import load_homography, pixel_to_world

# Config
base_x, base_y = 0, -11.8  # Base is 11.8 cm behind homography origin
l1, l2, l3 = 0, 12, 16  # Arm link lengths in cm
z_ground = 0.0  # Target z at table level
lift_height = 5.0  # cm above ground for lifting
z_lift = z_ground + lift_height
drop_positions = {
    "apple": (15, 0),  # Drop position for apple (in arm frame)
}

# Global flag to track pick sequence
is_pick_sequence = False

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

def print_command(joint, angle):
    """
    Simulate sending command to robot, applying offsets for pick sequence.
    """
    global is_pick_sequence
    adjusted_angle = angle
    
    # Apply offsets only during pick sequence
    if is_pick_sequence:
        if joint == "BASE":
            adjusted_angle = angle - 5
        elif joint == "SHOULDER":
            adjusted_angle = angle + 40
        elif joint == "ELBOW":
            adjusted_angle = angle - 40
    
    # Clip angles to joint limits
    if joint == "BASE":
        adjusted_angle = np.clip(adjusted_angle, 0, 180)
    elif joint == "SHOULDER":
        adjusted_angle = np.clip(adjusted_angle, 80, 160)
    elif joint == "ELBOW":
        adjusted_angle = np.clip(adjusted_angle, 0, 120)  # Changed to 0-120
    elif joint == "GRIP":
        adjusted_angle = np.clip(adjusted_angle, 0, 90)
    
    print(f"Would send: {joint}:{int(adjusted_angle)}")
    # For actual Arduino communication, uncomment:
    # import serial
    # ser = serial.Serial('COM3', 9600)  # Adjust port/baud
    # ser.write(f"{joint}:{int(adjusted_angle)}\n".encode())
    time.sleep(1)

def main():
    global is_pick_sequence
    model = load_model()
    H = load_homography()
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Webcam opened. Press 'p' to simulate pick-and-place for an apple, 'q' to quit.")

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
        elif keyboard.is_pressed('p') and len(det):
            for *xyxy, conf, cls in det:
                class_name = class_names[int(cls)].lower()
                if class_name != "apple":
                    continue
                
                center_u = (xyxy[0] + xyxy[2]) / 2
                center_v = (xyxy[1] + xyxy[3]) / 2
                real_x, real_y = pixel_to_world(H, center_u, center_v)
                print(f"Detected {class_name} at ({real_x:.2f}, {real_y:.2f}) cm")

                ik_x = real_x - base_x
                ik_y = real_y - base_y
                print(f"Adjusted position for IK: ({ik_x:.2f}, {ik_y:.2f}, {z_ground:.2f}) cm")

                # Pick: Move to z_ground, close gripper
                is_pick_sequence = True
                angles = inverse_kinematics(l1, l2, l3, ik_x, ik_y, z_ground)
                if angles:
                    base, shoulder, elbow = apply_joint_offsets(angles)
                    print_command("BASE", base)
                    print_command("SHOULDER", shoulder)
                    print_command("ELBOW", elbow)
                    print_command("GRIP", 0)
                    print_command("GRIP", 90)
                else:
                    print("Cannot reach pick position")
                    continue

                # Lift: Move to z_lift
                is_pick_sequence = False
                angles = inverse_kinematics(l1, l2, l3, ik_x, ik_y, z_lift)
                if angles:
                    base, shoulder, elbow = apply_joint_offsets(angles)
                    print_command("BASE", base)
                    print_command("SHOULDER", shoulder)
                    print_command("ELBOW", elbow)
                else:
                    print("Cannot reach lift position")
                    continue

                # Drop: Move to drop position at z_ground
                drop_x, drop_y = drop_positions[class_name]
                print(f"{class_name.capitalize()} detected! Moving to drop position ({drop_x}, {drop_y}, {z_ground})")
                angles = inverse_kinematics(l1, l2, l3, drop_x, drop_y, z_ground)
                if angles:
                    base, shoulder, elbow = apply_joint_offsets(angles)
                    print_command("BASE", base)
                    print_command("SHOULDER", shoulder)
                    print_command("ELBOW", elbow)
                    print_command("GRIP", 0)
                else:
                    print("Cannot reach drop position")
                    continue
                
                break

        cv2.waitKey(10)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()