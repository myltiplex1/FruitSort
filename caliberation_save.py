import cv2
import numpy as np

# Global variables for mouse clicks
pts_pixel = []
img = None

def mouse_callback(event, x, y, flags, param):
    global pts_pixel, img
    if event == cv2.EVENT_LBUTTONDOWN:
        pts_pixel.append([x, y])
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Image', img)
        print(f"Clicked: ({x}, {y})")

# Capture image
cap = cv2.VideoCapture(0)  # Confirmed index 1
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

ret, img = cap.read()
if not ret:
    print("Error: Cannot capture image")
    cap.release()
    exit()

cap.release()

# Markers
pts_world = np.array([[-10,0], [10,0], [-10,18], [10,18]], dtype='float32')

# Get pixel coordinates
print("Click the 4 markers in order: bottom-left (-10,0), bottom-right (10,0), top-left (-10,18), top-right (10,18)")
cv2.imshow('Image', img)
cv2.setMouseCallback('Image', mouse_callback)

while len(pts_pixel) < 4:
    cv2.waitKey(1)

pts_pixel = np.array(pts_pixel, dtype='float32')
print("Marker pixels:", pts_pixel)

# Compute homography
H, _ = cv2.findHomography(pts_pixel, pts_world)
np.save('homography.npy', H)
print("Homography saved as homography.npy")

# Function to map pixels to cm
def pixel_to_world(u, v):
    pt = np.array([u, v, 1]).T
    world_pt = np.dot(H, pt)
    world_pt /= world_pt[2]
    return world_pt[0], world_pt[1]

# Test point
print("Click the test point (e.g., fruit at (3.3,7.7) cm)")
pts_pixel = []
while len(pts_pixel) < 1:
    cv2.waitKey(1)

test_x, test_y = pts_pixel[0]
real_x, real_y = pixel_to_world(test_x, test_y)
print(f"Test point pixels: ({test_x}, {test_y})")
print(f"Mapped to cm: ({real_x:.2f}, {real_y:.2f})")

# For arm control
base_x, base_y = 0, -9
ik_x, ik_y = real_x - base_x, real_y - base_y
print(f"Arm input: ({ik_x:.2f}, {ik_y:.2f}, 0) cm")

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()