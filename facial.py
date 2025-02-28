import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

import os

directory = "Test_img"
IMAGE_FILES = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Initialize MediaPipe drawing utilities and face mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For static images:
# IMAGE_FILES = ['test_img.jpg']  # Replace with your image file path

# Initialize FaceMesh
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:

    for idx, file in enumerate(IMAGE_FILES):
        # Read the image
        image = cv2.imread(file)
        if image is None:
            print(f"Error: Unable to load image {file}. Check the file path.")
            continue

        # Convert the BGR image to RGB before processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        # Check if landmarks were detected
        if not results.multi_face_landmarks:
            print(f"No face landmarks detected in {file}.")
            continue

        # Create a matplotlib figure
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_image)

        # Plot the landmarks
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmark coordinates
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                landmarks.append((x, y))

            # Plot the landmarks as scatter points
            x_coords = [point[0] for point in landmarks]
            y_coords = [point[1] for point in landmarks]
            plt.scatter(x_coords, y_coords, c='red', s=5, alpha=0.5)  # Red dots for landmarks

        # Hide axes for better visualization
        plt.axis('off')
        plt.title(f"Facial Landmarks - {file}")
        plt.show()