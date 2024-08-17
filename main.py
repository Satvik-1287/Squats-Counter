import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)  # First point
    b = np.array(b)  # Vertex point
    c = np.array(c)  # End point
    
    # Vectors
    ab = a - b
    bc = c - b
    
    # Calculate the cosine of the angle
    cos_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure value is within valid range for arccos
    angle = np.degrees(np.arccos(cos_angle))
    
    return angle

def count_squats(landmarks):
    # Define the points for squat angle calculation
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

    # Calculate the angle at the knee
    angle = calculate_angle(
        (left_hip.x, left_hip.y),
        (left_knee.x, left_knee.y),
        (left_ankle.x, left_ankle.y)
    )

    # Define squat angle range
    squat_angle_threshold = 160  # Approximate angle for a squat position

    # Check if the angle is within the squat range
    if angle < squat_angle_threshold:
        return True
    return False

# Open video capture
cap = cv2.VideoCapture(0)
squat_count = 0
squat_started = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark

        # Check squat count
        if count_squats(landmarks):
            if not squat_started:
                squat_started = True
                squat_count += 1
        else:
            squat_started = False

    # Display squat count
    cv2.putText(frame, f'Squats: {squat_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Gym Tracker', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
