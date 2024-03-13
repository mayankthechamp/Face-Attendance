import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

jobs_image = face_recognition.load_image_file("Mayank1.jpg")
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]

ratan_tata_image = face_recognition.load_image_file("Mayank_photo.jpeg")
ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0]

sadmona_image = face_recognition.load_image_file("Abc.jpg")
sadmona_encoding = face_recognition.face_encodings(sadmona_image)[0]

tesla_image = face_recognition.load_image_file("cdf.jpg")
tesla_encoding = face_recognition.face_encodings(tesla_image)[0]

known_face_encoding = [jobs_encoding, ratan_tata_encoding, sadmona_encoding, tesla_encoding]
known_faces_names = ["jobs", "Mayank Singh", "sadmona", "tesla"]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.txt', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        name = ""
        face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        face_names.append(name)

        if name in students:
            students.remove(name)
            print(students)
            current_time = now.strftime("%H-%M-%S")
            lnwriter.writerow([name, current_time])

        # Display the name and attendance status on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 100)
        fontScale = 1.5
        fontColor = (255, 0, 0)
        thickness = 3
        lineType = 2
        cv2.putText(frame, f"{name} Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

    cv2.imshow("Attendance system", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
f.close()
