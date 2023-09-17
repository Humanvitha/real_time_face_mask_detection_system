from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import sys
import time
import RPi.GPIO as GPIO 
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
aMotorPins = [11,12,13,18] 
for pin in aMotorPins:
    GPIO.setup(pin,GPIO.OUT)
    GPIO.output(pin, False)
aSequence = [[1,0,0,1],[1,0,0,0],[1,1,0,0],[0,1,0,0],[0,1,1,0],[0,0,1,0],[0,0,1,1],[0,0,0,1]]        
iNumSteps = len(aSequence)
iDirection = 1
fWaitTime = 1 / float(1000)
iDeg = int(90 * 11.377777777777)
iSeqPos = 0
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        if mask > withoutMask:
            label = "Mask"
            color = (0, 255, 0)
            iDirection=1
            for step in range(0,iDeg):
                for iPin in range(0, 4):
                    iRealPin = aMotorPins[iPin]
                    if aSequence[iSeqPos][iPin] != 0:
                        GPIO.output(iRealPin, True)
                    else:
                        GPIO.output(iRealPin, False)
                iSeqPos += iDirection
                if (iSeqPos >= iNumSteps):
                    iSeqPos = 0
                if (iSeqPos < 0):
                    iSeqPos = iNumSteps + iDirection
                time.sleep(fWaitTime)
            for pin in aMotorPins:
                GPIO.output(pin, False)
            print('clockwise')
            iDirection = -1
            iSeqPos = 0
            for step in range(0,iDeg):
                for iPin in range(0, 4):
                    iRealPin = aMotorPins[iPin]
                    if aSequence[iSeqPos][iPin] != 0:
                        GPIO.output(iRealPin, True)
                    else:
                        GPIO.output(iRealPin, False)
                iSeqPos += iDirection
                if (iSeqPos >= iNumSteps):
                    iSeqPos = 0
                if (iSeqPos < 0):
                    iSeqPos = iNumSteps + iDirection
                time.sleep(fWaitTime)
            print('anti clock wise')
        else:
            label="No Mask"
            color=(0,0,255)
    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()

