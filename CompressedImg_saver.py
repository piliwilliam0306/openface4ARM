#! /usr/bin/python

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
from std_msgs.msg import String, Float64, Bool,UInt8
import os
import math
import time
# Instantiate CvBridge
bridge = CvBridge()
count = 0
transmit_progress = 0
path = 'data/mydataset/banana_aligned/jimmy'
images_required = 50.0
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
        	cv2.imwrite(os.path.join(path,'%s.jpeg' %banana), cv2_img)
    		image_progress = count / images_required * 100
		pub.publish(image_progress)
    else:
	return

def train_callback(msg):
    global count
    count = 0
    image_topic = "croppedImages/compressed"
    #rospy.Subscriber(image_topic, Image, image_callback)
    rospy.Subscriber(image_topic, CompressedImage, image_callback)
    #os.system('rm -rf /data/mydataset/banana_aligned/cache.t7')
    #while not rospy.is_shutdown():
    #	if count == images_required:
    #        os.system('./batch-represent/main.lua -outDir ./data/mydataset/banana_feature -data ./data/mydataset/banana_aligned')
    #	    break

def main():
    rospy.init_node('image_listener')
    
    train_topic = "cmdTrainning"
    rospy.Subscriber(train_topic, String, train_callback)
    #rospy.sleep(4)
    #rospy.loginfo("Finished taking %s images", %count)   
    #now = rospy.get_rostime()
    #while not rospy.is_shutdown():
    #	if count == 50:
    #		print "woolala"
    #		os.system('rm -rf /data/mydataset/banana_aligned/cache.t7') 
    #		os.system('./batch-represent/main.lua -outDir ./data/mydataset/banana_feature -data ./data/mydataset/banana_aligned')
    #new = rospy.get_rostime()
    #diff = new.secs - now.secs
    #rospy.loginfo("Generate Representations took %i seconds", diff)

    #os.system('./demos/classifier.py train ./data/mydataset/banana_feature')  
    #new2 = rospy.get_rostime()
    #diff = new2.secs - new.secs
    #rospy.loginfo("Create the Classification Model %i seconds", diff)

    #rospy.loginfo("Training done")
    #rospy.signal_shutdown("Done.") 
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    if not os.path.exists(path):
    	os.makedirs(path)
    pub = rospy.Publisher('capturingProgress', UInt8, queue_size=1)
    main()
