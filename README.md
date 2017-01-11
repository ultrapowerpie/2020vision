# 2020vision

Timeline/Outline:
https://docs.google.com/document/d/1nhJT8y2f3HaJnzlTRZRcVAg9sct1xv9ixvtth-2T_ZM/edit?ts=5859c697

Final Report: https://docs.google.com/a/princeton.edu/document/d/1p2AxmUXlTm5fseO_sGOsRoSoYja-aJtcgY_Ew9_8cq8/edit?usp=sharing


# Running VGG16 pre-trained model:
1. Download pre-trained weights and put in root: https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc
2. Run with "python run_vgg16.py [path to file]
3. Outputs index of category in training data (from authors)

# Running with run.py
1. Be in the 2020vision Directory
2. You can choose training parameters in the run.py file (try not to change the
    random see, and if you do, record the changes.)
3. $ python run.py <arg1>
    <arg1> can be "basic_v1", "basic_v2", "fancy_v1", "fancy_v2"


# Net structures

VGG_16 - We used the 3x3 convlutional filters from VGG

ResNet - We used residual layers (2-layer skip connections) in the fancy nets

ZFNet - We used the 

Convolution-only Net

Basic_Net:

This network is composed of blocks of 3x3 convolution layers
(as proposed by VGG), but with 2x2 subsampling on each layer instead of pooling
as proposed by the Conovolution-only nets paper.

It still uses the traditional convolution then pooling structure as in in LeNet
instead of VGG's multiple convolution blocks.

V2 simply has double the channels.

Fancy_Net:

This network has a similar structure to Basic_Net, but uses residual layers
over every two convolutional layers, and the two-conv-layer block is composed of
one non-subsampled convolution layer followed by a pooled on, similar to the
tradional ResNet/VGG structure.

The V2 version uses the identity residual blocks that connect between
convolution to batch normalization, rather than activation to another
convolution, and may train slightly faster and converge more stablely.

# Training

I was able to train up to 128x128 images on all 4 nets, but Basic_Net_v2 still
seems to give the best results in terms of top-1 accuracy at around 72% after
10 epochs.

Using K-Nearest Neighbors (with 5 neighbors), top-1 accuracy for Basic_Net_v2
increased from 72% to 74.6%. Have not yet tested on Fancy_Net_v2 or for other
numbers of neighbors. Prediciton takes a long time because model must be
run on entire training set to build intermediate output.
