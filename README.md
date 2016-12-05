
# Prerequisite (have openface and torch installed)

##Go to openface workspace
    $ cd ~/openface/models/openface

##Downloading nn4.v1.t7 model
    $ wget http://openface-models.storage.cmusatyalab.org/nn4.v1.ascii.t7.xz
    $ unxz nn4.v1.ascii.t7.xz

##Downloading nn4.small2.v1.t7 model
    $ wget http://openface-models.storage.cmusatyalab.org/nn4.small2.v1.ascii.t7.xz
    $ unxz nn4.small2.v1.ascii.t7.xz

##Convert model from ascii to binary
    $ th
    th> require 'nn'
    th> require 'dpnn'
    th> net = torch.load('nn4.v1.ascii.t7', 'ascii')
    torch.save('nn4.v1.t7', net)
    th> net = torch.load('nn4.small2.v1.ascii.t7', 'ascii')
    torch.save('nn4.small2.v1.t7', net)

##Solve out of memory problem when installing torch on TK1 
    modify torch/extra/cutorch/rocks/cutorch-scm-1.rockspec by replacing "-j$(getconf _NPROCESSORS_ONLN)" with "-j1".

##Test if odroid can get images from android
    $ rosrun openface4ARM CompressedImg_saver.py

##Setting label
    $ rostopic pub /cmdTrainning std_msgs/String "data: 'yo'"
<!--
##Running openface_ros node
    $ rosrun openface4ARM openface_server.py train ./data/mydataset/banana_feature
    
##Tranning service (replace banana with new member name, done msg will be dumped when finish training)
    $ rosservice call /trainning "name: 'banana'"
![](https://github.com/piliwilliam0306/openface4ARM/blob/master/train.jpg)

##Recognition service (member will be recognized)
    $ rosservice call /whoami "request: 'yo'"
![](https://github.com/piliwilliam0306/openface4ARM/blob/master/infer.jpg)
-->
# Reference
https://cmusatyalab.github.io/openface/setup

https://github.com/cmusatyalab/openface/issues/42

https://github.com/torch/torch7/blob/master/doc/serialization.md

https://www.ottoii.com/2016/08/14/94#.WAh36xJ96ao

https://github.com/mlennox/tk1-torch-install
