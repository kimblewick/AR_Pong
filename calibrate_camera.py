import cv2
import numpy as np
import os
import glob
import time

# --- Configuration ---
CHESSBOARD_SIZE = (9, 6) # Number of inner corners (width-1, height-1)
SQUARE_SIZE_METERS = 0.025 # Size of one chessboard square in meters (e.g., 2.5 cm)
MIN_IMAGES = 15 # Minimum number of images required for calibration
CALIBRATION_FILENAME = "calibration_data.npz"
CAMERA_INDEX = 1 # Change if your camera is not index 0

# --- Prepare Object Points ---
# 3D points in real world space (origin at one corner, Z=0)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_METERS # Scale to meters

# --- Arrays to store object points and image points from all images ---
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane

# --- Setup Camera ---
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
    exit()

print("\n--- Camera Calibration ---")
print(f"Using chessboard size: {CHESSBOARD_SIZE}")
print(f"Square size: {SQUARE_SIZE_METERS * 1000} mm")
print(f"Need at least {MIN_IMAGES} valid images.")
print("""
Instructions:
- Show the chessboard to the camera.
- Move the board to different positions and angles.
- Ensure corners are detected (drawn on screen).
- Press SPACE to capture an image.
- Press Q when you have captured >= {MIN_IMAGES} images to calibrate.
""")

captured_images = 0
last_capture_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret_corners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    display_frame = frame.copy() # Draw on a copy

    # If found, add object points, image points (after refining them)
    if ret_corners:
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw the corners
        cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners_subpix, ret_corners)

        key = cv2.waitKey(1) & 0xFF

        # Capture image on SPACE press (with a small delay between captures)
        if key == ord(' ') and time.time() - last_capture_time > 0.5:
            print(f"Image {captured_images + 1} captured.")
            objpoints.append(objp)
            imgpoints.append(corners_subpix)
            captured_images += 1
            last_capture_time = time.time()
            # Flash screen green briefly
            display_frame[:,:,1] = 255


        # Calibrate and Quit on Q press (if enough images)
        elif key == ord('q'):
            if captured_images >= MIN_IMAGES:
                print(f"\nCaptured {captured_images} images. Starting calibration...")
                break # Exit loop to perform calibration
            else:
                print(f"Need at least {MIN_IMAGES} images, only have {captured_images}. Press SPACE to capture more.")

    else:
         key = cv2.waitKey(1) & 0xFF
         if key == ord('q'):
             print("\nExiting without calibration.")
             cap.release()
             cv2.destroyAllWindows()
             exit()


    # --- Display Status ---
    cv2.putText(display_frame, f"Captured: {captured_images}/{MIN_IMAGES}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if ret_corners:
         cv2.putText(display_frame, "Press SPACE to capture", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "Show Chessboard", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(display_frame, "Press Q to Calibrate/Quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow('Camera Calibration - Press SPACE to capture, Q to finish', display_frame)


# --- Perform Calibration ---
print("Calculating camera matrix and distortion coefficients...")
try:
    ret_cal, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret_cal:
        print("\nCalibration successful!")
        print("\nCamera Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)

        # --- Save Calibration Data ---
        print(f"\nSaving calibration data to {CALIBRATION_FILENAME}...")
        np.savez(CALIBRATION_FILENAME, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        print("Calibration data saved.")

        # --- Calculate Reprojection Error ---
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        reprojection_error = mean_error / len(objpoints)
        print(f"\nTotal Reprojection Error: {reprojection_error:.4f} pixels")
        print("(Lower is better, ideally < 1.0)")

    else:
        print("\nCalibration failed!")

except Exception as e:
    print(f"\nAn error occurred during calibration: {e}")


# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("\nCalibration script finished.") 