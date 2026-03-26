import cv2
import numpy as np
import face_recognition # Fixed: removed 's'
import os
from datetime import datetime

path = 'Images_Attendance'
images = []
classNames = []
myList = os.listdir(path)
print(f'Found images: {myList}')

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
print(f'Class Names: {classNames}')

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        # Safety check: only append if at least one face is found
        if len(encodings) > 0:
            encodeList.append(encodings[0])
        else:
            print("Warning: A training image had no detectable face and was skipped.")
    return encodeList

def markAttendance(name):
    # Ensure the file exists before reading/writing
    if not os.path.exists('Attendance.csv'):
        with open('Attendance.csv', 'w') as f:
            f.writelines('Name,Time,Date')

    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        
        if name not in nameList:
            now = datetime.now()
            tString = now.strftime('%H:%M:%S')
            dString = now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tString},{dString}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Resizing for faster processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        if len(faceDis) > 0:
            matchIndex = np.argmin(faceDis)

            # Threshold check (0.6 is default, lower is stricter)
            if matches[matchIndex] and faceDis[matchIndex] < 0.6:
                name = classNames[matchIndex].upper()
                
                # Rescale coordinates back to original image size
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

    cv2.imshow('Webcam', img)
    # Press 'Enter' to exit
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows() # Fixed: Added the 's' to Window
