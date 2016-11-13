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
path = 'data/mydataset/banana_aligned/banana'

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
    #rospy.loginfo("Finished taking %s images", %count)   
    now = rospy.get_rostime()
    #os.system('rm -rf /data/mydataset/banana_aligned/cache.t7') 
    os.system('./batch-represent/main.lua -outDir ./data/mydataset/banana_feature -data ./data/mydataset/banana_aligned')
    new = rospy.get_rostime()
    diff = new.secs - now.secs
    rospy.loginfo("Generate Representations took %i seconds", diff)

    os.system('./demos/classifier.py train ./data/mydataset/banana_feature')  
    new2 = rospy.get_rostime()
    diff = new2.secs - new.secs
    rospy.loginfo("Create the Classification Model %i seconds", diff)

    rospy.loginfo("Training done")
    rospy.signal_shutdown("Done.") 
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    if not os.path.exists(path):
    	os.makedirs(path)
    
    main()
