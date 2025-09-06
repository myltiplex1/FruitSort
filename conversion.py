import numpy as np

def load_homography(file='homography.npy'):
    H = np.load(file)
    return H

def pixel_to_world(H, u, v):
    pt = np.array([u, v, 1]).T
    world_pt = np.dot(H, pt)
    world_pt /= world_pt[2]
    return world_pt[0], world_pt[1]

if __name__ == "__main__":
    H = load_homography()
    print("Enter a test pixel coordinate (e.g., from your homography setup)")
    u = float(input("Pixel u (x): "))
    v = float(input("Pixel v (y): "))
    x, y = pixel_to_world(H, u, v)
    print(f"Pixel ({u}, {v}) maps to world ({x:.2f}, {y:.2f}) cm")