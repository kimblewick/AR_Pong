import cv2
import numpy as np
from cv2 import aruco
import mediapipe as mp
import time
import os # Import os for file path checking

# --- Camera Calibration --- #
calibration_file = "calibration_data.npz"

# Default/Placeholder values (used if calibration file not found)
default_camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
default_dist_coeffs = np.zeros((5, 1), dtype=np.float32)

if os.path.exists(calibration_file):
    try:
        with np.load(calibration_file) as data:
            camera_matrix = data['camera_matrix']
            dist_coeffs = data['dist_coeffs']
            print(f"Loaded camera calibration from {calibration_file}")
    except Exception as e:
        print(f"Error loading calibration file {calibration_file}: {e}")
        print("Using default calibration values.")
        camera_matrix = default_camera_matrix
        dist_coeffs = default_dist_coeffs
else:
    print(f"Calibration file {calibration_file} not found.")
    print("Using default placeholder calibration values - AR accuracy will be limited!")
    camera_matrix = default_camera_matrix
    dist_coeffs = default_dist_coeffs


# --- Chessboard Setup ---
CHESSBOARD_SIZE = (9, 6) 
SQUARE_SIZE_METERS = 0.025 

# Prepare theoretical 3D object points for the chessboard corners
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_METERS # Scale to meters

# --- Table Dimensions (in meters) ---
# Assume origin is roughly center of board for table placement.
TABLE_WIDTH_METERS = 0.4 # Example width (40 cm)
TABLE_HEIGHT_METERS = 0.6 # Example height (60 cm)

# --- Game State ---
surface_detected = False
game_running = False
rvec = None # Rotation vector of the surface
tvec = None # Translation vector of the surface
table_corners_2d = None # To store projected 2D corners
perspective_matrix = None # Transformation matrix for normalized coords -> image coords
inverse_perspective_matrix = None # Transformation matrix for image coords -> normalized coords
score = 0
ball_pos = np.array([0.5, 0.5]) # Normalized 2D coordinates on the table [0,1]
ball_vel = np.array([0.01, 0.01]) # Normalized velocity
paddle_pos = 0.5 # Normalized position [0,1] along the player edge
start_time = time.time()

# --- Hand Tracking Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Helper Function ---
def get_image_coords(norm_x, norm_y, M):
    """Maps normalized coordinates (0-1) to image coordinates using perspective transform M."""
    if M is None:
        return None
    norm_point = np.array([[[norm_x, norm_y]]], dtype=np.float32)
    image_point = cv2.perspectiveTransform(norm_point, M)
    if image_point is None or image_point.size == 0:
        return None
    return tuple(image_point[0][0].astype(int))

def get_normalized_coords(img_x, img_y, inv_M):
    """Maps image coordinates to normalized coordinates (0-1) using inverse perspective transform inv_M."""
    if inv_M is None:
        return None
    img_point = np.array([[[float(img_x), float(img_y)]]], dtype=np.float32)
    norm_point = cv2.perspectiveTransform(img_point, inv_M)
    if norm_point is None or norm_point.size == 0:
        return None
    return tuple(norm_point[0][0])

# --- Main Loop ---
cap = cv2.VideoCapture(1) # Use 1 for iPhone camera (Continuity Camera on Mac)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Find Chessboard Corners ---
    ret_corners, corners_found = cv2.findChessboardCorners(frame_gray, CHESSBOARD_SIZE, None)

    # Reset detection status for this frame
    surface_detected_this_frame = False
    target_rvec, target_tvec = None, None

    if ret_corners: 
        # --- Pose Estimation using Chessboard corners ---
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(frame_gray, corners_found, (11, 11), (-1, -1), criteria)

        retval, rvec, tvec = cv2.solvePnP(objp, corners_subpix, camera_matrix, dist_coeffs) # CHESSBOARD

        if retval:
            surface_detected_this_frame = True
            target_rvec, target_tvec = rvec, tvec
         
            cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners_subpix, ret_corners) # CHESSBOARD

            # Draw relative to the board center (approx) for consistency
            board_center_3d = np.mean(objp, axis=0).reshape(1,1,3)
            axis_points_3d = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,-0.1]]).reshape(-1,3) # 10cm axes
            # Use solvePnP result (rvec, tvec) which defines board pose relative to camera
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1) # Draw 10cm axes at board origin

    else:
        pass # No corners detected

    # --- Update Game State based on Target Marker Detection ---
    if surface_detected_this_frame:
        surface_detected = True
        rvec, tvec = target_rvec, target_tvec # Use the pose from the target marker

        # --- Define and Project Table ---
        half_width = TABLE_WIDTH_METERS / 2
        half_height = TABLE_HEIGHT_METERS / 2

        table_corners_3d = np.array([
            [-half_width, half_height, 0], # Top-Left
            [ half_width, half_height, 0], # Top-Right
            [ half_width,-half_height, 0], # Bottom-Right
            [-half_width,-half_height, 0]  # Bottom-Left
        ], dtype=np.float32)

        # Project the 3D corners onto the 2D image plane
        projected_corners, _ = cv2.projectPoints(table_corners_3d, rvec, tvec, camera_matrix, dist_coeffs)

        if projected_corners is not None:
            table_corners_2d = np.int32(projected_corners.reshape(-1, 2))

            # Calculate perspective transform matrix
            src_points = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]) # Top-L, Top-R, Bot-R, Bot-L
            # Ensure table_corners_2d order matches src_points order
            dst_points = table_corners_2d.astype(np.float32)
            perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            try:
                inverse_perspective_matrix = np.linalg.inv(perspective_matrix)
            except np.linalg.LinAlgError:
                print("Warning: Could not compute inverse perspective matrix (singular?).")
                inverse_perspective_matrix = None
                perspective_matrix = None # If inverse fails, forward probably unusable too
                table_corners_2d = None

            # Draw the table outline
            cv2.polylines(frame, [table_corners_2d], isClosed=True, color=(0, 255, 255), thickness=2)

            # Draw center line
            p1 = tuple((table_corners_2d[0] + table_corners_2d[3]) // 2) # Midpoint of left edge
            p2 = tuple((table_corners_2d[1] + table_corners_2d[2]) // 2) # Midpoint of right edge
            cv2.line(frame, p1, p2, (0, 255, 255), 1)

        else:
            # Failed to project even if marker detected?
            table_corners_2d = None
            perspective_matrix = None
            inverse_perspective_matrix = None

    else: # Target marker NOT detected this frame
        surface_detected = False
        game_running = False # Stop game if target detection lost
        # Reset projection variables
        rvec, tvec = None, None
        table_corners_2d = None
        perspective_matrix = None
        inverse_perspective_matrix = None

    # --- Hand Tracking and Paddle Control ---
    if surface_detected and inverse_perspective_matrix is not None: # Need inverse transform for control
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Optional: Draw landmarks

                # Use the wrist position to control the paddle
                wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                h, w, c = frame.shape
                wrist_px = int(wrist_landmark.x * w)
                wrist_py = int(wrist_landmark.y * h)

                # --- Draw Hand Bounding Box ---
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for landmark in hand_landmarks.landmark:
                    px, py = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, px)
                    y_min = min(y_min, py)
                    x_max = max(x_max, px)
                    y_max = max(y_max, py)
                # Add some padding
                padding = 20
                cv2.rectangle(frame, (x_min - padding, y_min - padding), (x_max + padding, y_max + padding), (0, 255, 0), 2)
                # --- End Hand Bounding Box ---

                # Convert wrist image coordinates to normalized table coordinates
                norm_coords = get_normalized_coords(wrist_px, wrist_py, inverse_perspective_matrix)

                if norm_coords is not None:
                    norm_x, norm_y = norm_coords
                    # Use the normalized x-coordinate to control the paddle position
                    paddle_pos = norm_x

                    # Clamp paddle position (keep within normalized bounds, e.g., 0.05 to 0.95 to prevent going fully off-edge)
                    paddle_width_norm = 0.15 # Example size
                    paddle_limit_lower = paddle_width_norm / 2
                    paddle_limit_upper = 1.0 - paddle_width_norm / 2
                    paddle_pos = max(paddle_limit_lower, min(paddle_limit_upper, paddle_pos))

        # --- Game Logic ---
        if not game_running:
            # Simple start condition: Wait a bit after detection
            if time.time() - start_time > 2: # Wait 2 seconds after detection
                 game_running = True
                 # TODO: Initialize ball position/velocity properly
                 ball_pos = np.array([0.5, 0.8]) # Start near player
                 ball_vel = np.array([np.random.uniform(-0.01, 0.01), -0.015]) # Initial velocity
                 score = 0


        if game_running:
             # Update ball position
             ball_pos += ball_vel

             # Collision Detection (Normalized Coordinates [0, 1])
             table_width_norm = 1.0
             table_height_norm = 1.0
             ball_radius_norm = 0.03 # Example size
             paddle_width_norm = 0.15 # Example size
             paddle_height_norm = 0.05 # Example thickness

             # Side walls
             if ball_pos[0] <= ball_radius_norm or ball_pos[0] >= table_width_norm - ball_radius_norm:
                 ball_vel[0] *= -1
                 ball_pos[0] = np.clip(ball_pos[0], ball_radius_norm, table_width_norm - ball_radius_norm) # Prevent sticking

             # Top wall
             if ball_pos[1] <= ball_radius_norm:
                 ball_vel[1] *= -1
                 ball_pos[1] = ball_radius_norm # Prevent sticking
                 score += 1
                 # Increase speed (progressive difficulty)
                 ball_vel *= 1.05

             # Paddle collision (Bottom wall - player)
             # Check if ball is near the player's edge
             if ball_pos[1] >= table_height_norm - paddle_height_norm - ball_radius_norm:
                 # Check if ball aligns horizontally with the paddle
                 paddle_left = paddle_pos - paddle_width_norm / 2
                 paddle_right = paddle_pos + paddle_width_norm / 2
                 if paddle_left <= ball_pos[0] <= paddle_right:
                     # Collision!
                     # Reverse Y direction
                     ball_vel[1] = -abs(ball_vel[1]) # Ensure it moves away from paddle
                     # Add random horizontal deflection (scaled by current speed magnitude)
                     current_speed = np.linalg.norm(ball_vel)
                     deflection = np.random.uniform(-0.3, 0.3) # Random factor
                     ball_vel[0] += deflection * current_speed * 0.3 # Adjust scaling factor as needed
                     # Clamp max horizontal speed if needed?
                     ball_pos[1] = table_height_norm - paddle_height_norm - ball_radius_norm # Move ball back slightly
                     # Optional: Add effect based on paddle impact location/speed

             # Ball missed (went past player edge)
             if ball_pos[1] > table_height_norm + ball_radius_norm: # Allow some margin
                 # Reset ball (or end game)
                 game_running = False # For now, just stop the game on miss
                 print(f"Game Over! Final Score: {score}")
               


             # --- Rendering Game Elements ---
             if perspective_matrix is not None: # Ensure we have the transform matrix

                 # Calculate ball position in image coordinates
                 ball_img_coords = get_image_coords(ball_pos[0], ball_pos[1], perspective_matrix)

                 # Calculate paddle position in image coordinates
                 paddle_y_norm = 1.0 - paddle_height_norm / 2 # Use paddle center line Y
                 paddle_left_norm = paddle_pos - paddle_width_norm / 2
                 paddle_right_norm = paddle_pos + paddle_width_norm / 2

                 paddle_img_start = get_image_coords(paddle_left_norm, paddle_y_norm, perspective_matrix)
                 paddle_img_end = get_image_coords(paddle_right_norm, paddle_y_norm, perspective_matrix)

                 # Draw ball
                 if ball_img_coords is not None:
                     # Scale ball radius roughly based on perspective (average side length?)
                     # Basic approach: fixed pixel radius for now
                     ball_radius_pixels = 8
                     cv2.circle(frame, ball_img_coords, ball_radius_pixels, (0, 0, 255), -1)

                 # Draw paddle
                 if paddle_img_start is not None and paddle_img_end is not None:
                     cv2.line(frame, paddle_img_start, paddle_img_end, (0, 0, 255), 5)


    # --- Display Info ---
    cv2.putText(frame, f"Surface Detected: {surface_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if surface_detected else (0, 0, 255), 2)
    if game_running:
        cv2.putText(frame, f"Score: {score}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("AR Hockey Pong", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Application exited.") 