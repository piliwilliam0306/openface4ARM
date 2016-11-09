#! /usr/bin/python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from std_msgs.msg import String
import os
# Instantiate CvBridge
bridge = CvBridge()
count = 0
transmit_progress = 0
path = 'data/mydataset/raw/banana'

def image_callback(msg):
    global count
    count = count + 1
    if count < 61:	
    	rospy.loginfo("Received %s images!" %count) 
    	try:
        	# Convert your ROS Image message to OpenCV2
        	cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    	except CvBridgeError, e:
        	print(e)
    	else:
        	# Save your OpenCV2 image as a jpeg
        	#cv2.imwrite('banana%s.jpeg' %count, cv2_img)
        	cv2.imwrite(os.path.join(path,'banana%s.jpeg' %count), cv2_img)
    else:
	return

def main():
    rospy.init_node('image_listener')
    
    # Define your image topic
    image_topic = "/banana/image"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    rospy.sleep(4)
    rospy.loginfo("Finished taking images, start cropping")   
    now = rospy.get_rostime()

    os.system('for N in {1..4}; do ./util/align-dlib.py ./data/mydataset/raw align outerEyesAndNose ./data/mydataset/banana_aligned --size 96 & done')  
    new2 = rospy.get_rostime()
    diff = new2.secs - now.secs
    rospy.loginfo("Cropping took %i seconds", diff)

    rospy.loginfo("Cropping done")
    rospy.signal_shutdown("Done.") 
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    if not os.path.exists(path):
    	os.makedirs(path)
    main()
    #os.system('./batch-represent/main.lua -outDir ./data/mydataset/banana_feature -data ./data/mydataset/banana_aligned') 
    #rospy.loginfo("feature done") 
    #os.system('./demos/classifier.py train ./data/mydataset/banana_feature')
    #main()
