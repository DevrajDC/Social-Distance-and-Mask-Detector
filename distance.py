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
from scipy.spatial import distance as dist


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
    cv2.putText(frame, "Social Distance Detector", (130, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
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
                # playsound("beep-01a.wav")
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
            cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif i in abnormal:
            text = 'Normal'
            cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
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
