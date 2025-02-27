import cv2
from anime_face_detector import create_detector

# Load the anime face detector
detector = create_detector('yolov3')

# Load the image
image_path = 'test_img.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Detect faces in the image
faces = detector(image)

# Draw the face contours
for face in faces:
    # Get the bounding box coordinates
    x1, y1, x2, y2 = face['bbox']
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Optional: Draw facial landmarks (if available)
    if 'landmarks' in face:
        for landmark in face['landmarks']:
            x, y = landmark
            cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)

# Save or display the result
output_path = 'output_image.jpg'
cv2.imwrite(output_path, image)
print(f"Result saved to {output_path}")

# Display the image (optional)
cv2.imshow('Anime Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()