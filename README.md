
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
    th> net = torch.load('small2.v1.ascii.t7', 'ascii')
    torch.save('nn4.small2.v1.t7', net)

# Reference
https://cmusatyalab.github.io/openface/setup

https://github.com/cmusatyalab/openface/issues/42

https://github.com/torch/torch7/blob/master/doc/serialization.md
