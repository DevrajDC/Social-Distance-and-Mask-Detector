from tkinter import *
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from PIL import Image, ImageTk
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from mylib import config, thread
from mylib.mailer import Mailer
from mylib.detection import detect_people
from imutils.video import FPS
from playsound import playsound
from scipy.spatial import distance as dist


def masked():
    def detect_and_predict_mask(frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            preds = maskNet.predict(faces)
            mask_accuracy = preds[0][0]
            no_mask_accuracy = preds[0][1]

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)

    # construct the argument parser and parse the arguments
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

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(args["model"])

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


def Social_distance():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="",
                    help="path to (optional) input video file")
    ap.add_argument("-o", "--output", type=str, default="",
                    help="path to (optional) output video file")
    ap.add_argument("-d", "--display", type=int, default=1,
                    help="whether or not output frame should be displayed")
    args = vars(ap.parse_args())

    labelsPath = "coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    weightsPath = 'yolov4-tiny.weights'
    configPath = 'yolov4-tiny.cfg'
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # check if we are going to use GPU
    if config.USE_GPU:
        # set CUDA as the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # if a video path was not supplied, grab a reference to the camera
    if not args.get("input", False):
        print("[INFO] Starting the live stream..")
        vs = cv2.VideoCapture(0)
        if config.Thread:
            cap = thread.ThreadingClass(config.url)
        time.sleep(2.0)

    # otherwise, grab a reference to the video file
    else:
        print("[INFO] Starting the video..")
        vs = cv2.VideoCapture(args["input"])
        if config.Thread:
            cap = thread.ThreadingClass(args["input"])

    writer = None
    # start the FPS counter
    fps = FPS().start()

    # loop over the frames from the video stream
    while True:
        # read the next frame from the file
        if config.Thread:
            frame = cap.read()

        else:
            (grabbed, frame) = vs.read()
            # if the frame was not grabbed, then we have reached the end of the stream
            if not grabbed:
                break

        # resize the frame and then detect people (and only people) in it
        cv2.putText(frame, "Social Distance Detector", (130, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln,
                                personIdx=LABELS.index("person"))
        # initialize the set of indexes that violate the max/min social distance limits
        serious = set()
        abnormal = set()
        # ensure there are at least two people detections (required in
        # order to compute our pairwise distance maps)
        if len(results) >= 2:
            # extract all centroids from the results and compute the
            # Euclidean distances between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # check to see if the distance between any two
                    # centroid pairs is less than the configured number
                    # of pixels
                    if D[i, j] < config.MIN_DISTANCE:
                        # update our violation set with the indexes of
                        # the centroid pairs
                        serious.add(i)
                        serious.add(j)
                        # playsound("Final/face-mask-detection/beep-01a.wav")
                    # update our abnormal set if the centroid distance is below max distance limit
                    if (D[i, j] < config.MAX_DISTANCE) and not serious:
                        abnormal.add(i)
                        abnormal.add(j)

        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract the bounding box and centroid coordinates, then
            # initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            # if the index pair exists within the violation/abnormal sets, then update the color
            if i in serious:
                text = "Red Alert"
                color = (0, 0, 255)
                cv2.putText(frame, text, (startX, startY - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif i in abnormal:
                text = 'Normal'
                cv2.putText(frame, text, (startX, startY - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                color = (0, 255, 0)  # orange = (0, 165, 255)

            # draw (1) a bounding box around the person and (2) the
            # centroid coordinates of the person,
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        # cv2.circle(frame, (cX, cY), 5, color, 2)

        # draw some of the parameters
        # Safe_Distance = "Safe distance: >{} px".format(config.MAX_DISTANCE)
        # cv2.putText(frame, Safe_Distance, (470, frame.shape[0] - 25),
        # 	cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
        # Threshold = "Threshold limit: {}".format(config.Threshold)
        # cv2.putText(frame, Threshold, (470, frame.shape[0] - 50),
        # 	cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)

        # draw the total number of social distancing violations on the output frame
        text = "Social Distance Violation: {}".format(len(serious))
        cv2.putText(frame, text, (10, frame.shape[0] - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2)

        text1 = "Safe Count: {}".format(len(abnormal))
        cv2.putText(frame, text1, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ------------------------------Alert function----------------------------------#
        if len(serious) >= config.Threshold:
            cv2.putText(frame, "-ALERT: Violations over limit-", (10, frame.shape[0] - 80),
                        cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 2)
            if config.ALERT:
                print("")
                print('[INFO] Sending mail...')
                Mailer().send(config.MAIL)
                print('[INFO] Mail sent')
        # config.ALERT = False
        # ------------------------------------------------------------------------------#
        # check to see if the output frame should be displayed to our screen
        if args["display"] > 0:
            # show the output frame
            cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        # update the FPS counter
        fps.update()

        # if an output video file path has been supplied and the video
        # writer has not been initialized, do so now
        if args["output"] != "" and writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 25,
                                     (frame.shape[1], frame.shape[0]), True)

        # if the video writer is not None, write the frame to the output video file
        if writer is not None:
            writer.write(frame)

    # stop the timer and display FPS information
    fps.stop()
    print("===========================")
    print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

    # close any open windows
    cv2.destroyAllWindows()


root = Tk()
root.title("Project")
root.geometry('800x600')
img_file = Image.open('im2.png')
bg = ImageTk.PhotoImage(img_file)
bgl = Label(root, image=bg)
bgl.place(x=0, y=0)
# bg = PhotoImage('img.jpg')
# my_canvas = Canvas(root, width=800, height=500)
# my_canvas.pack(fill="both", expand=True)
#
# # Set image in canvas
# my_canvas.create_image(0, 0, image=bg, anchor="nw")
#
# # Add a label
# my_canvas.create_text(400, 250, text="Team Vision!", font=("Helvetica", 50), fill="white")
lbl = Label(root, text="Team vision", font=('Helvetica', 25, 'underline'))
lbl.pack()
bt2 = Button(root, text="Face mask Detection", bd=8, command=masked)
bt2.place(x=60, y=300)
bt1 = Button(root, text='Social Distance detector',
             bd=8, command=Social_distance)
bt1.place(x=600, y=300)
root.resizable(0, 0)
root.mainloop()
