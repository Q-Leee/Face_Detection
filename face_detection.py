import cv2
import sys

# Use OpenCV's internal data path for Haar cascades
cascPath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread("E:\\Python\\Face_Detection\\image\\test3.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

# Print the number of faces detected
print("Found {0} faces!".format(len(faces)))

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue rectangle

# Display the output image with rectangles
cv2.imshow("Faces found", image)

# Keep the window open until a key is pressed
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()