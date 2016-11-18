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
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

#faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from openface4ARM.srv import *
bridge = CvBridge()

def image_callback(msg):
        rospy.loginfo("Received images!")
        try:
                # Convert your ROS Image message to OpenCV2
                #cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
                cv2_img = bridge.imgmsg_to_cv2(msg, "rgb8")
        except CvBridgeError, e:
                print(e)
	start = time.time()
	bb = align.getAllFaceBoundingBoxes(cv2_img)
	print("dlib Found {0} faces!".format(len(bb)))
	print("Face detection took {} seconds.".format(time.time() - start))
	alignedFaces = []
    	for box in bb:
            alignedFaces.append(
            	align.align(
                	args.imgDim,
                	#rgbImg,
			cv2_img,
                	box,
                	landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))
	reps = []
    	for alignedFace in alignedFaces:
        	reps.append(net.forward(alignedFace))
	with open(args.classifierModel, 'r') as f:
            (le, clf) = pickle.load(f)  # le - label and clf - classifer
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
    	print "P: " + str(persons) + " C: " + str(confidences)
	#string.data = persons
	pub.publish(data=str(persons))

if __name__ == '__main__':
    rospy.init_node('people_rec_stream')
    image_topic = "/camera/rgb/image_raw"
    rospy.Subscriber(image_topic, Image, image_callback)
    pub = rospy.Publisher('member', String, queue_size=10)
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
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
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

    confidenceList = []
    try:
    	#try:
            # append with two floating point precision
        #    confidenceList.append('%.2f' % confidences[0])
    	#except:
            # If there is no face detected, confidences matrix will be empty.
            # We can simply ignore it.
        #    pass

    	#for i, c in enumerate(confidences):
        #    if c <= args.threshold:  # 0.5 is kept as threshold for known face.
        #        persons[i] = "_unknown"

                # Print the person name and conf value on the frame
        #cv2.putText(frame, "P: {} C: {}".format(persons, confidences),
        #            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        

	#cv2.imshow('', frame)
        # quit the program on the press of key 'q'
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
	rospy.spin()
        #try:
  	#    rospy.spin()
        #except KeyboardInterrupt:
  	#    print("Shutting down")
  	#    cv2.destroyAllWindows()
    except KeyboardInterrupt:
            print("Shutting down")
    # When everything is done, release the capture
    #video_capture.release()
    #cv2.destroyAllWindows()
