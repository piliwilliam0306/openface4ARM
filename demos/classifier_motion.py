#!/usr/bin/env python2
#
# Example to run classifier on webcam stream.
# Brandon Amos & Vijayenthiran
# 2016/06/21
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Contrib: Vijayenthiran
# This example file shows to run a classifier on webcam stream. You need to
# run the classifier.py to generate classifier with your own dataset.
# To run this file from the openface home dir:
# ./demo/classifier_webcam.py <path-to-your-classifier>


import time

start = time.time()

import argparse
import cv2
import os
import pickle

import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
import openface
import dlib
win = dlib.image_window()

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
trackingFace = 0
fps = 0
motion_detected = False
lastFrame = None

def getRep(bgrImg):
    global trackingFace
    global motion_detected
    start = time.time()
    if bgrImg is None:
        raise Exception("Unable to load image/frame")

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if args.verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    # Get the largest face bounding box
    #bb = align.getLargestFaceBoundingBox(rgbImg) #Bounding box
    # Get all bounding boxes
    if not trackingFace:
    	bb = align.getAllFaceBoundingBoxes(rgbImg)
        if bb is not None: 
	    try:
	        tracker.start_track(rgbImg,dlib.rectangle(bb[0].left(),bb[0].top(),bb[0].right(),bb[0].bottom()))
            except:
	        return ([],None)	
	    trackingFace = 1
	else:
	    motion_detected = False
	    print("unable to detect your face, please face the camera")
    else:
	trackingQuality = tracker.update(rgbImg)
	if trackingQuality >= 8.75:
            bb1 = tracker.get_position()	
            bb1 = dlib.rectangle(int(bb1.left()),int(bb1.top()),int(bb1.right()),int(bb1.bottom()))
	    bb = [bb1]
	else:
	    trackingFace = 0
	    motion_detected = False
	    print("unable to detect your face, please face the camera")
	    return ([],None)

    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))
    start = time.time()

    alignedFaces = []
    for box in bb:
        alignedFaces.append(
            align.align(
                args.imgDim,
                rgbImg,
                box,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

    if alignedFaces is None:
        raise Exception("Unable to align the frame")
    if args.verbose:
        print("Alignment took {} seconds.".format(time.time() - start))

    start = time.time()

    reps = []
    for alignedFace in alignedFaces:
        reps.append(net.forward(alignedFace))

    if args.verbose:
        print("Neural network forward pass took {} seconds.".format(
            time.time() - start))

    return (reps, bb)

def infer(img, args):
    global motion_detected
    with open(args.classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)  # le - label and clf - classifer

    reps, bb = getRep(img)
    persons = []
    confidences = []
    for rep in reps:
        try:
            rep = rep.reshape(1, -1)
        except:
            print "No Face detected"
            return (None, None)
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        # print predictions
        maxI = np.argmax(predictions)
        # max2 = np.argsort(predictions)[-3:][::-1][1]
        persons.append(le.inverse_transform(maxI))
        # print str(le.inverse_transform(max2)) + ": "+str( predictions [max2])
        # ^ prints the second prediction
        confidences.append(predictions[maxI])
        if args.verbose:
            print("Prediction took {} seconds.".format(time.time() - start))
            pass
        # print("Predict {} with {:.2f} confidence.".format(person, confidence))
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
            pass
    return (persons, confidences, bb)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        default=os.path.join(
            openfaceModelDir,
            'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument(
        '--captureDevice',
        type=int,
        default=0,
        help='Capture device. 0 for latop webcam and 1 for usb webcam')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        'classifierModel',
        type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')

    args = parser.parse_args()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(
        args.networkModel,
        imgDim=args.imgDim,
        cuda=args.cuda)
    tracker = dlib.correlation_tracker()
    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    video_capture = cv2.VideoCapture(args.captureDevice)
    video_capture.set(3, args.width)
    video_capture.set(4, args.height)

    confidenceList = []
    t_start = time.time()
    while True:
        ret, frame = video_capture.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
	
	avg_area = align.motionDetect(gray, lastFrame)
	if args.verbose:
	    print(avg_area)
	if ((avg_area < 500) & (trackingFace == 0)):
            motion_detected = False
        else:
            motion_detected = True

	if motion_detected == True:
            persons, confidences, bb = infer(frame, args)
	else:
	    persons, confidences, bb = ([], [], None)

	fps = fps + 1
	sfps = fps / ( time.time() - t_start )
        print "P: " + str(persons) + " C: " + str(confidences) + "FPS: " + str(sfps)
	print("motion state: {}".format(motion_detected))
	if bb is not None:
            try:
                # append with two floating point precision
                confidenceList.append('%.2f' % confidences[0])
	        x = bb[0].left()
                y = bb[0].top()
                w = bb[0].right()
                h = bb[0].bottom()
            except:
                # If there is no face detected, confidences matrix will be empty.
                # We can simply ignore it.
                pass

            for i, c in enumerate(confidences):
                if c <= args.threshold:  # 0.5 is kept as threshold for known face.
                    persons[i] = "_unknown"

                # Print the person name and conf value on the frame
            cv2.putText(frame, "P: {} C: {} FPS: {}".format(persons, confidences, sfps),
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    #(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 3)	
	cv2.imshow('', frame)
	lastFrame = gray
        # quit the program on the press of key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
