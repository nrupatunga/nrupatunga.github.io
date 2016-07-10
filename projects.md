---
layout: page
permalink: /project/
title: Projects Code & Software
---

##**Pedestrian Detection using Histogram of Oriented Gradients ( HOG )**
HOG is a visual descriptor i.e., it describes the content of an image in a single feature vector. 
The idea behind HOG is that local object appearance and shape within an image can be described by 
the distribution of intensity gradients or edge directions.
{% include image.html url="./hog/hog-descriptor-calculation.PNG" description="HOG descriptor " %}

As shown in the figure, we first compute the gradient map.
HOG descriptor is calculated for a window size of 64x128, by dividing 
it into 8x16 cells and in each cell we calculate the orientation of all
pixels and form a 9-bin histogram of gradients. These gradients are
normalized by overlapping block size of 2x2 cells. Finally we
concatenate all the orientations into a single vector of length 3780.
For more deeper understanding of HOG, please refer to this nice
[tutorial](http://mccormickml.com/2013/05/09/hog-person-detector-tutorial/)

<sup>_NOTE: Here numerical values are taken to explain the mathematics of
HOG calculation. These values can be varied while computation_ </sup>

I trained soft margin linear SVM model on the dataset with 4419 positive samples and 5380
negative samples. I did two iterations of **_hard negative mining_** to improve
the results and also performed **_non-maximum suppression_** to filter the
detection windows.
{% include image.html url="./hog/pedestriandetection.PNG" description="Pedestrian Detection " %}

<sup>**Code:** [C++, Python](https://github.com/nrupatunga/Pedestrain-Detection-using-Histogram-of-Oriented-Gradients)</sup>
<br> 

##**Global Image Descriptor - GIST**
GIST is the low dimensional representation of an Image. It encodes the
structural information of the image by dividing the image into blocks,
thus providing a rough description of the image.
The block diagram of how GIST is computed is shown below. 
For more theory, refer to this [paper](http://people.csail.mit.edu/torralba/code/spatialenvelope/). 
{% include image.html url="./gist/gistblockdiagram.png" description="GIST block diagram" %}

In order to visualize how GIST feature could encode the information of an
image, I projected the 512-dimensional GIST feature vector to a
2-dimensional space using dimensionality reduction technique called
[t-SNE](https://lvdmaaten.github.io/tsne/) and generated t-SNE
visualization using the [code](http://cs.stanford.edu/people/karpathy/cnnembed/) provided by Andrej Karpathy.

{% include image.html url="./gist/t-sne/gist-nn-small.png" description="t-SNE visualization" %}
Please [download](./gist/t-sne/gist-nn-large.png) and zoom into different parts of the image and see how similar images are clustered together.
Thus GIST features helps in the task of Nearest Neighbor Image retrieval

<sup>**Tool:** [Command line tool](https://github.com/nrupatunga/GIST-global-Image-Descripor)</sup>
&nbsp; &nbsp; &nbsp; &nbsp;
&nbsp; &nbsp; &nbsp; &nbsp;
&nbsp; &nbsp; &nbsp; 
<sup>**Code:** [C++](https://github.com/nrupatunga/GIST-global-Image-Descripor)</sup>
<br> 

##**Pencil Sketch**
This is the implementation of the algorithm in this [paper](http://www.cse.cuhk.edu.hk/~leojia/projects/pencilsketch/npar12_pencil.pdf). 
It is a new system to produce pencil drawings from natural images. 
This method mimicks the human style of Pencil Drawing by combining tonal
and sketch structure.

Input           |  Pencil Sketch
:-------------------------:|:-------------------------:|
![](./Pencil/In-1.jpg)  | ![](./Pencil/outputgraysketch.png)

 Color Pencil Sketch |
:-------------------------:|
![](./Pencil/outputcolorsketch.png)  |

<sup>**Software:**  [Windows Executable](https://github.com/nrupatunga/Pencil-Sketch/releases) 
&nbsp; &nbsp; &nbsp; &nbsp;
&nbsp; &nbsp; &nbsp; &nbsp;
&nbsp; &nbsp; &nbsp; 
**Demo Video:**  [Pencil Sketch](https://nrupatunga-gmail.tinytake.com/sf/NzQwOTk5XzM0MTEzOTM)
&nbsp; &nbsp; &nbsp; &nbsp;
&nbsp; &nbsp; &nbsp; &nbsp;
&nbsp; &nbsp; 
**Code:** [C++](https://github.com/nrupatunga/Color-Pencil-Sketch)</sup>
<br> 

<sup>_**Note:**_ _Software does not include color pencil sketch for now!_

##**Image Processing Toolbox**
This is a GUI application developed using OpenCV and Qt. This
application can be used to experiment the following functionalities.

* Edge Detection
	- Sobel
	- Canny
* Blur
	- Homogeneous
	- Median
	- Gaussian
	- Bilateral

<sup>**Software:**  [Windows Executable](https://github.com/nrupatunga/Computer-Vision-Tool) 
&nbsp; &nbsp; &nbsp; &nbsp;
&nbsp; &nbsp; &nbsp; &nbsp;
&nbsp; &nbsp; &nbsp; 
&nbsp; &nbsp; &nbsp; &nbsp;
&nbsp; &nbsp; &nbsp; &nbsp;
**Demo Video:**  [Image Processing Toolbox](https://nrupatunga-gmail.tinytake.com/sf/NzUyNTg4XzM0NDE0OTM)
<br> 

##**Bilateral Filtering**
A [bilateral filter](http://people.csail.mit.edu/sparis/bf_course/) is a non-linear, edge-preserving and noise-reducing smoothing filter for images. 

Input           |  Gaussian Filtering | Bilateral Filtering
:-------------------------:|:-------------------------:|:----------------------:|
![](./BilateralFilter/Input.jpg)  | ![](./BilateralFilter/GaussOutput.jpg) | ![] (./BilateralFilter/BilateralOutput.jpg )
<sup>**Code:** [C++](https://github.com/nrupatunga/Bilateral-Filter)<br></sup>

##**Canny Edge Detector**
The [Canny edge detector](http://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html#gsc.tab=0) is an edge detection 
operator that uses a multi-stage algorithm to detect a wide range of edges in images

Input           |  Canny Edge Output 
:-------------------------:|:-------------------------:|
![](./Canny/flower.jpg)  | ![](./Canny/cannyedge.jpg)
<sup>**Code:** [C++](https://github.com/nrupatunga/Canny-Edge-Detector)</sup>
<br>

##**Python Learning**
I have been coding in Python for a while. You can find a very good
introductory book on Python by [Swaroop] (http://python.swaroopch.com/)  which introduces different concepts of
Python language very well. 

I have made a IPython notebook while practising the code given in this book. You
can check it out in the below github link. I constantly refer to this
when I am coding. 

<sup>**Code:** [IPython Notebook](https://github.com/nrupatunga/Learning-Python)</sup>
