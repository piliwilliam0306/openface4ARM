#!/usr/bin/env python2
#
# Example to classify faces.
# Brandon Amos
# 2015/10/11
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

import time

start = time.time()

import argparse
import cv2
import os
import pickle

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
#ros wrapper
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, UInt16, Float64, Bool,UInt8
from openface4ARM.srv import *
bridge = CvBridge()
count = 0

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
pickleDir = os.path.join(fileDir, 'data/mydataset/banana_feature')
recDir = 'data/mydataset/banana_rec'
featureDir = 'data/mydataset/banana_feature'
path = ''
images_required = 0.0
transmit_progress = 0
dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
networkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
imgDim = 96

def image_callback(msg):
    global count
    count = count + 1
    banana = time.strftime("%H:%M:%S")
    if count < images_required + 1:
        rospy.loginfo("Received %s images!" %count)
        try:
                # Convert your ROS Image message to OpenCV2
                #cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
		cv2_img = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError, e:
                print(e)
        else:
                # Save your OpenCV2 image as a jpeg
		cv2.imwrite(os.path.join(path,'IMAGE%s.jpeg' %count), cv2_img)
                #cv2.imwrite(os.path.join(path,'%s.jpeg' %banana), cv2_img)
		image_progress = count / images_required *100
                pub.publish(image_progress)
    else:
        return

def getRep(imgPath):#, multiple=False):
    start = time.time()
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    #if args.verbose:
    #    print("  + Original size: {}".format(rgbImg.shape))
    #if args.verbose:
    #    print("Loading the image took {} seconds.".format(time.time() - start))

    #start = time.time()

    #if multiple:
    #    bbs = align.getAllFaceBoundingBoxes(rgbImg)
    #else:
    bb1 = align.getLargestFaceBoundingBox(rgbImg)
    bbs = [bb1]
    #if len(bbs) == 0 or (not multiple and bb1 is None):
    #    raise Exception("Unable to find a face: {}".format(imgPath))
    #if args.verbose:
    #    print("Face detection took {} seconds.".format(time.time() - start))

    reps = []
    for bb in bbs:
        start = time.time()
        alignedFace = align.align(
            imgDim,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))
        #if args.verbose:
        #    print("Alignment took {} seconds.".format(time.time() - start))
        #    print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))

        #start = time.time()
	###comment out to ignore align face
        rep = net.forward(alignedFace)
        #rep = net.forward(rgbImg)
        #if args.verbose:
        #    print("Neural network forward pass took {} seconds.".format(
        #        time.time() - start))
        reps.append((bb.center().x, rep))

    #rep = net.forward(rgbImg)
    #reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps

def train_callback(msg):
    global path
    global count 
    global images_required
    images_required = 50.0
    count = 0
    path = ('data/mydataset/banana_aligned/{}/{}'.format(msg.data,msg.data))
    if not os.path.exists(path):
        os.makedirs(path)

    #os.system('rm data/mydataset/banana_aligned/cache.t7')
    while not rospy.is_shutdown():
    	if count == images_required:
	    now = rospy.get_rostime()
	    #os.system('./batch-represent/main.lua -outDir ./data/mydataset/banana_feature -data ./data/mydataset/banana_aligned')
	    #os.system('./batch-represent/main.lua -outDir ./data/mydataset/banana_feature -data ./data/mydataset/banana_aligned/{}'.format(msg.data))
	    #test.lua is for register new member only
	    os.system('./batch-represent/test.lua -outDir ./data/mydataset/banana_feature -data ./data/mydataset/banana_aligned/{}'.format(msg.data))
	    new = rospy.get_rostime()
	    diff = new.secs - now.secs
	    rospy.loginfo("Feature generation took %i seconds", diff)
	    print("Loading embeddings.")
	    #fname = "{}/labels.csv".format(args.workDir)
	    fname = "{}/labels.csv".format(featureDir)
	    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
	    labels = map(itemgetter(1),
		         map(os.path.split,
		             map(os.path.dirname, labels)))  # Get the directory.
	    #fname = "{}/reps.csv".format(args.workDir)
	    fname = "{}/reps.csv".format(featureDir)
	    embeddings = pd.read_csv(fname, header=None).as_matrix()
	    le = LabelEncoder().fit(labels)
	    labelsNum = le.transform(labels)
	    nClasses = len(le.classes_)
	    print("Training for {} classes.".format(nClasses))

	    clf = SVC(C=1, kernel='linear', probability=True)

	    clf.fit(embeddings, labelsNum)

	    #fName = "{}/classifier.pkl".format(args.workDir)
	    fName = "{}/classifier.pkl".format(featureDir)
	    print("Saving classifier to '{}'".format(fName))
	    with open(fName, 'w') as f:
		pickle.dump((le, clf), f)
	    pub1.publish(100)
	    break

def rec_callback(msg):
    if msg.data == True:
        global path
        global images_required
        global count
        path = 'data/mydataset/banana_rec'
        images_required = 1.0
        count = 0
        #with open(args.classifierModel, 'r') as f:
        with open(os.path.join(pickleDir,'classifier.pkl'), 'r') as f:
        #f == os.path.join(pickleDir,'classifier.pkl')
            (le, clf) = pickle.load(f)
        while not rospy.is_shutdown():
	    if count == images_required:
	        #for img in recDir:
	        img = "data/mydataset/banana_rec/IMAGE1.jpeg"
		print("\n=== {} ===".format(img))
		reps = getRep(img)#, multiple)
		if len(reps) > 1:
		    print("List of faces in image from left to right")
		for r in reps:
		    rep = r[1].reshape(1, -1)
		    bbx = r[0]
		    start = time.time()
		    predictions = clf.predict_proba(rep).ravel()
		    maxI = np.argmax(predictions)
		    person = le.inverse_transform(maxI)
		    confidence = predictions[maxI]
		    #if args.verbose:
		    #    print("Prediction took {} seconds.".format(time.time() - start))
		    #if multiple:
		    #    print("Predict {} @ x={} with {:.2f} confidence.".format(person, bbx,
		    #                                                                 confidence))
		    #else:
		    print("Predict {} with {:.2f} confidence.".format(person, confidence))
		    if isinstance(clf, GMM):
		        dist = np.linalg.norm(rep - clf.means_[maxI])
		        print("  + Distance from the mean: {}".format(dist))
	    	    pub2.publish(person)
	        break


if __name__ == '__main__':
    if not os.path.exists(recDir):
        os.makedirs(recDir)
    rospy.init_node('people_rec')
    pub = rospy.Publisher('capturingProgress', UInt8, queue_size=1)
    pub1 = rospy.Publisher('trainingProgress', UInt8, queue_size=1)
    pub2 = rospy.Publisher('recognitionResults', String, queue_size=1)
    train_topic = "cmdTraining"
    rec_topic = "cmdRecognition"
    image_topic = "croppedImages/compressed"
    rospy.Subscriber(image_topic, CompressedImage, image_callback)
    rospy.Subscriber(train_topic, String, train_callback)    
    rospy.Subscriber(rec_topic, Bool, rec_callback)

    start = time.time()

    align = openface.AlignDlib(dlibFacePredictor)
    net = openface.TorchNeuralNet(networkModel, imgDim=imgDim)#,
    #                              cuda=args.cuda)
    """	
    if args.verbose:
        print("Loading the dlib and OpenFace models took {} seconds.".format(
            time.time() - start))
        start = time.time()
    """
    rospy.spin() 
