from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import argparse
import imutils
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time as t
from firebase import firebase
import board
import adafruit_mlx90614
import busio as io



path = 'att_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

# get people names from the dataset file
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# get faces encodings from the dataset
def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList



# a function for mask detection
def detect_and_predict_mask(frame, faceNet, maskNet):
    #shape the frame of the image
    print(type(frame))
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    #obtaining face detection
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    # loop over the the dataset of mask detections
    for i in range(0, detections.shape[2]):
        
        #get the accuracy level of mask detection compared to dataset 
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            
            # calculate the bounding box dimensions
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # resize and process
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # if face mask is detected
    if len(faces) > 0:
        preds = maskNet.predict(faces)

    # return the face location
    return (locs, preds)


def Attendance(name):

    #record the attendance in a firebase realtime database
    fbcon = firebase.FirebaseApplication('https://attendance-caa4d-default-rtdb.firebaseio.com/', None)
    now = datetime.now()
    time = now.strftime('%d/%m/%Y - %H:%M')
    temp = 36.5 #btemp
    data = {
        'Name': name,
        'Time': time,
        'Temperature': temp
    }
    result = fbcon.post('/attendance/', data)
    print(result)
    t.sleep(1)

    #record data in excel file for easy access
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            f.writelines(f'\n{name}, {time}, {temp}')




# arguments for mask detections
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load models from the dataset
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model(args["model"])


encodeListKnown = find_encodings(images)
print("Encoding completed")

# initializing video output
cap = cv2.VideoCapture(0)
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
t.sleep(2.0)

while True:
    
    # grab the frame from the video and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    print(type(cap))

    # detect faces and determine if they are wearing a face mask
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # drawing the bounding box for mask detection
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    
    #drawing the bounding box for face recognition
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
    # capture the face from via the camera and check for matches in the dataset
    for encodeFace, faceloc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6,), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            
            #temperature sensor (i2c is a predefined address)
            i2c = io.I2C(board.SCL, board.SDA, frequency=100000)
            mlx=adafruit_mlx90614.MLX90614(i2c)
            btemp= mlx.object_temperature
            print("\nbody temperature is ", btemb)
            Attendance(name)


    #display the video stream on the screen
    cv2.imshow('Webcam', img)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
