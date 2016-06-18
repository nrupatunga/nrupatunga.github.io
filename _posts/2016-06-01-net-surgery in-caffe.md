---
layout: post
permalink: /netsurgery/
---

<sub>
**Note**: Please download the [IPython notebook](../assets/net_surgery_fcn.ipynb) from this link.
</sub>

Table of Contents:


* [Introduction](#L1)
* [Network architecture](#L2)
* [Weight transplant](#L3)

<a name="L1"></a>
# Introduction

This is a hands on converting a Convolutional Neural Network (CNN) to Fully Convolution Network (FCN). Converting CNN to FCN is nothing but 
converting fully connected (FC) layers in CNN to convolution layers 

Let's take the standard Caffe Reference ImageNet model [CaffeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet) and
transform it into a fully convolutional net. The code is extracted and modified from the cafee [net surgery](https://github.com/BVLC/caffe/blob/master/examples/net_surgery.ipynb) example

<a name="L2"></a>
# Network architecture
Before converting FC layers to convolution layers, lets see the network architecture first. 
Figure 1 shows the architecture of CaffeNet.

<sub> (Open the image in new tab and zoom to see the details) </sub>
{% include image.html url="../assets/net-surgery/bvlc.png" description="Figure 1: CaffeNet architecture" %}
####Python Code
{% include image.html url="../assets/net-surgery/architecture-caffenet.png" description="" %}
#### Output
{% include image.html url="../assets/net-surgery/outputarch.png" description="" %}

* Execute the python code, output prints the layers in **_CaffeNet_** and dimensions of weights in each
layer. Here for the sake of simplicity I have eliminated bias parameters. As you can see, it has 5 convolution 
layers (_conv1, conv2, conv3, conv4, conv5_) and three FC (_fc6, fc7, fc8_) layers

* Lets understand the weight dimensions format in convolution and FC layers. Figure 2 is self explanatory.
{% include image.html url="../assets/net-surgery/filter.png" description="Figure 2: Weight dimensions in Convolution and FC layers" %}

* Now that we know the network architecture and the weight dimensions in each layer, our intension is to convert FC (_fc6, fc7, fc8_) layers to convolution layers

* As explained in Figure 2, in dimensions of _fc6_ layer \\((4096, 9216)\\), \\(9216\\) indicates the number of outputs from convolution layer-5 after pooling. 
Lets just confirm that.
{% include image.html url="../assets/net-surgery/pool5.png" description="Figure 3: Output dimensions from convolution layer-5 after pooling" %}

* Observe \\(256 * 6 * 6\\) = \\(9216\\), so in order to convert _fc6_ layer to convolution layer, we just need to use the convolution kernel of size 6. 
The respective prototxt file change is shown in Figure 4.
{% include image.html url="../assets/net-surgery/fc6-conv6.png" description="Figure 4: fc6 to convolution layer" %}

* The rest of fully connected layers _fc7_, _fc8_ can be viewed as convolution layer with \\(1\\) x \\(1\\) kernel.
The respective prototxt file change is shown in Figure 5.
{% include image.html url="../assets/net-surgery/fc78.png" description="Figure 5: fc7, fc8 to convolution layer" %}

* Lets verify it now.

####Python Code
{% include image.html url="../assets/net-surgery/fc78-python.png" description="" %}
#### Output
{% include image.html url="../assets/net-surgery/fc78-output.png" description="" %}

<a name="L3"></a>
# Weight transplant
Now that we converted the network architecture, lets transfer the
weights from CNN to FCN and generate the classification map.

####Python Code
{% include image.html url="../assets/net-surgery/class-python.png" description="" %}
#### Output
{% include image.html url="../assets/net-surgery/class-output.png" description="" %}
As you can see in the probability map values, the classifications include various cats -- 282 = tiger cat, 281 = tabby, 283 = persian.

So FCN can be used to extract dense feature maps. This enables us dense
learning (eg. Image Semantic Segmenation).

That is it, we have converted CNN to FCN. It is easy isn't it?.
