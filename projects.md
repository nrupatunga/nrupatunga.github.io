---
layout: page
permalink: /project/
title: Projects & Software
---

##**Global Image Descriptor - GIST**
GIST is the low dimensional representation of an Image. It encodes the
structural information of the image by dividing the image into blocks,
thus providing a rough description of the image.
The block diagram of how GIST is computed is shown below. 
For more theory, refer to this [paper](http://people.csail.mit.edu/torralba/code/spatialenvelope/). 
{% include image.html url="./gist/gistblockdiagram.png" description="" %}
<sup>**Tool:** [Command line tool](https://github.com/nrupatunga/GIST-global-Image-Descripor)</sup>
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
&nbsp; &nbsp; &nbsp;
**Code** [C++](https://github.com/nrupatunga/Color-Pencil-Sketch)</sup>
<br> 

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
&nbsp; &nbsp;
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
