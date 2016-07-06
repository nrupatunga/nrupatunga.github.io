---
layout: post
permalink: /convolution-1/
---

Table of Contents:

*   [Introduction](#L1)
*   [Affine Transformation and Discrete convolution](#L2)
*   [Discrete Convolution in detail](#L3)
*   [Convolutional Arithmetic](#L4)
	- [No Zero Padding, Unit Strides](#L41)
	- [Zero Padding, Unit Strides](#L42)
	- [Half Padding](#L43)
	- [Full Padding](#L44)
	- [No Zero Padding, non-unit Strides](#L45)
	- [Zero Padding, non-unit Strides](#L46) 
* [References](#LR)

<a name="L1"></a>
## Introduction

Convolutional Neural Networks ([CNN]
(https://en.wikipedia.org/wiki/Convolutional_neural_network)) has
become so popular due to its state of the art results in many computer
vision tasks such as _Image Recognition_, _Image Classification_,
_Semantic Segmentation_.

For a beginner in Deep Learning, using CNNs for the first time is
generally an intimidating experience. Even though it is easier to
understand the layers of CNNs such as _Convolution_,
_Pooling_, _Non Linear Activations_, _Fully Connected layers_,
_Deconvolution_ when treated individually, its quite difficult to make 
sense of effect of these operations on each layer's output size, shape 
as the networks gets deeper

[Figure 1.1](http://arxiv.org/pdf/1505.04366v1.pdf) below shows the
sample CNN architecture. We can see how the network is stacked up with
_Convolution_, _Pooling_, _Deconvolution_ layers. At the top of each
layers, you can observe the shape of the output mentioned. How did we
get those numbers?. This is dependent on the parameters choosen in that layer.
That's what we want to understand.
{% include image.html url="../assets/cnn.jpg" description="Figure 1.1: CNN architecture" %}

I experienced that good understanding of computational mechanism of convolution, pooling and 
deconvolution layers and their dependency on parameters such as _kernel size, strides, padding_ 
together builds a solid ground in understanding CNNs better

**_The main objectives for rest of the post:_**

* To understand the relationship between input shape, kernel shape, zero
    padding, strides and output shape in convolution, pooling and
    deconvolution layers
* To understand the relationship between convolution layers and
    deconvolution layers

The main building blocks of CNN are **_Discrete Convolution_** and
**_Pooling_**. 

<a name="L2"></a>
## Affine Transformation and Discrete Convolution

* **_Affine Transformation_**: A vector is given as an input and is
    multiplied with a matrix to produce an output. This is the
    transformation which is most often used in Neural networks. This is
    applicable to any type of input be it an image, a sound clip:
    whatever their dimensionality, the representation can always be
    flattened into a vector before transformation
	- If you take [MNIST](http://yann.lecun.com/exdb/mnist/) digit recognition as an example, the
	 2-D input is flattened to a 1-D vector of size and fed as an input to
	 the typical [fully connected neural network]
	 (http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/) as shown in Figure 1.2
	    {% include image.html url="../assets/mnist.png" description="Figure 1.2: Multi-layer perceptron" %}
	- But we notice images, sound clips have intrinsic structure. They share following properties
		- They are stored as multi-dimensional array
		- They feature one or more axes for which ordering
		    matters (e.g., width and height axes of an image,
		    time axis for a sound clip)
		- One axis, called the channel axis is used to access
		    different views of the data (e.g., red, green, and
		    blue channels of a color image, or left and right
		    channels of a stereo audio track)
	- These properties are not exploited when _Affine transformation_ is applied. All the axes are treated
		  in the same way and topological information is not taken into consideration
	- In computer vision tasks, taking the advantage of implicit structure of the input data can be very handy

* **_Discrete Convolution_**: It is a linear transformation which respects the ordering of the input data that we discussed above
	- Convolution operation is sparse in CNN, i.e., only a few input units contribute to a
	    given output unit (Figure 1.3 (a))
	- It reuses the parameters i.e., same weights are applied to
	    multiple locations (Figure 1.3 (b))
	    {% include image.html url="../assets/localconnectivity_sharedweights.png" description="Figure 1.3: (a) Local connectivity (b) Shared weights" %}


<a name="L3"></a>
## Discrete Convolution in detail
* Figure 1.4 shows an example of a discrete convolution. The light blue
    grid is called the **_input feature map_** and the shaded area is the
    **_kernel_** (_kernel values are at right bottom of each cell in shaded area_)
    {% include image.html url="../assets/convolution-2.1.png" description="Figure 1.4: Computing the output values of a discrete convolution" %}
* At each location, the product between the each element of the kernel
    and the input elements which overlaps is computed and results are
    summed up to obtain the output in the current location. The green
    grid in the figure illustrates this, shaded area in the green grid
    indicates output at that current location
* The convolution operation shown in the Figure 1.4 is an instance of
    2-D convolution, but this can be generalized to N-D convolution. 
    For instance, in a 3-D convolution, the kernel would be a
    _cuboid_ and would slide across the height, width and the depth of
    the input.
* In CNN, the collection of kernels defining a discrete convolution has a shape
    corresponding to some permutation of (\\(n, m, k\_j\\)),
    where 
    $$ n \equiv number\  of\ output\ feature\ maps$$
    $$ m \equiv number\  of\ input\ feature\ maps$$
    $$ k\_j \equiv kernel\  size\ along\ axis\ j \ (\ width \ or \ height \ axis\ )$$
* The following properties affect the output size \\(o\_j\\) of a
    convolution layer along the axis \\(j\\)
    $$ i\_j : input\  size\ along\ axis\ j$$
    $$ k\_j : kernel\  size\ along\ axis\ j$$
    $$ s\_j : stride\  along\ axis\ j$$
    $$ p\_j : zero\  padding\ along\ axis\ j$$

* For instance Figure 1.5 shows a \\(3\*3\\) kernel applied to
    \\(5\*5\\) input padded with \\(1\*1\\) border of zeros using \\(2\*2\\)
    strides
{% include image.html url="../assets/convolution-2.2.png" description="Figure 1.5: Computing the output values of a discrete convolution for 
	  \\(N=2\\)  \\(i\_1 = i\_2 = 5\\),  \\(k\_1 = k\_2 = 3\\) ,  \\(s\_1 = s\_2 = 2\\),  \\(p\_1 = p\_2 = 1\\)" %}

* The strides constitute a form of **_subsampling_**. Strides can be
    viewed as much of the output is retained.  For instance, moving the
    kernel by hops of two is equivalent to moving the kernel by hops
    of one, but retain only odd output elements (Figure 1.6)
{% include image.html url="../assets/convolution-2.3.png" description="Figure 1.6: An alternative way of viewing strides. Instead of 
translating the \\(3\*3\\) kernel by increments of \\(s=2\\) (left), the kernel is translated by increments of \\(1\\) and only odd numbered output elements are retained" %}

<a name="L4"></a>
## Convolutional Arithmetic
The analysis of relationship between convolutional layer properties is
eased by the fact that they don't interact across axes, i.e., the choice
of kernel size, stride and zero padding along the axis \\(j\\) only affects
the output size along the axis \\(j\\)

The following simplified settings are used to analyse the convolution
layer properties

* \\(2\\)-\\(D\\) discrete convolutions (\\(N = 2\\))
* square inputs (\\(i\_1 = i\_2 = i\\))
* square kernel size (\\(k\_1 = k\_2 = k\\))
* same strides along both axes (\\(s\_1 = s\_2 = s\\))
* same zero padding along both axes (\\(p\_1 = p\_2 = p\\))

<a name="L41"></a>
### No Zero Padding, Unit Strides
The simplest case to analyse is when the kernel just slides across every position of the input (i.e., \\(s = 1\\) and \\(p = 0\\))
{% include image.html url="../assets/no_padding_no_strides.png" description="Figure 1.7: No Padding, Unit Strides" %}

Lets define the output size resulting from this setting

* The output size is the number of possible placements of the
    kernel on the input
* Lets consider the width axis: the kernel starts on the
    leftmost part of the input feature map and slides by steps
    of one until it touches the right side of the input
* The size of the output will be equal to number of steps made,
    plus one

**Relationship-1**: _For any \\(i\\), \\(k\\), and for \\(s=1\\) and \\(p=0\\),_

$$o = (i-k)+1$$

* Figure 1.7 provides an example for \\(i=4\\), \\(k = 3\\), \\(s=1\\), therefore output size \\(o = (4-3) + 1 = 2\\) along each axis

<a name="L42"></a>
### Zero Padding, Unit Strides
Lets consider zero padding only restricting stride \\(s = 1\\). The effect of
zero padding increases the size of the input from \\(i\\) to \\(i+2p\\)
{% include image.html url="../assets/arbitrary_padding_no_strides.PNG" description="Figure 1.8: Zero Padding, Unit Strides" %}

**Relationship-2**: _For any \\(i\\), \\(k\\), \\(p\\) and for \\(s=1\\),_
$$o = (i-k)+ 2p + 1$$

* Figure 1.8  provides an example for \\(i = 5\\), \\(k = 4\\) and \\(p = 2\\),
    therefore output size \\(o = (5-4) + 2*2 + 1 = 6\\).

<a name="L43"></a>
### Half padding 	
Some times we require the output size of convolution to be same as input
size (i.e., \\(o=i\\)). In order for \\(o=i\\), we use \\(p = \lfloor \dfrac{k}{2} \rfloor\\)
{% include image.html url="../assets/same_padding_no_strides.PNG" description="Figure 1.9: Half Padding, Unit Strides" %}

**Relationship-3**: For any \\(i\\), and for odd \\(k\\) (\\(k = 2n + 1\\), \\(n \in N\\)), \\(s=1\\), and \\(p = \lfloor \dfrac{k}{2} \rfloor = n\\) 
$$o = (i+2\lfloor \dfrac{k}{2} \rfloor)-(k - 1)$$
$$o = (i + 2n - 2n)$$
$$o = i$$

* Figure 1.9  provides an example for \\(i = 5\\), \\(k = 3\\) hence \\(p = 2\\),
    therefore output size \\(o = (5+2*\lfloor \dfrac{3}{2} \rfloor) - (3 - 1) = 5 + 2 * 1 - 2 = 5\\).

<a name="L44"></a>
### Full padding 	
Some times we require the output size of convolution to be of larger size than as input. But, convolution always decreases 
the size of the output if there is no extra padding to the input, so we can do some extra zero padding to the input.
{% include image.html url="../assets/full_padding_no_strides.PNG" description="Figure 1.10: Full Padding, Unit Strides" %}
**Relationship-4**: For any \\(i\\), and \\(k\\), \\(s=1\\), and \\(p = k - 1\\) 
$$o = (i + 2(k - 1) - (k - 1)$$
$$o = (i + (k - 1)$$

* Figure 1.10  provides an example for \\(i = 5\\), \\(k = 3\\) hence \\(p = 2\\),
    therefore output size \\(o = 5 + 3 - 1 = 7\\).

<a name="L45"></a>
### No zero padding, non-unit strides
All relationships which we saw till now are unit-strided convolutions. In order to understand the effect of non-unit strides, 
lets ignore padding for now.

{% include image.html url="../assets/no_padding_strides.PNG" description="Figure 1.11: No zero padding, non-unit Strides" %}
* As we discussed before, output size can be defined in terms of the number of possible placements of the kernel on the input.
If you consider only width axis, the size of the output is equal to the number of steps made, plus one, accounting for the initial position of the kernel.
The same logic applies for height axis.

**Relationship-5**: For any \\(i\\), \\(k\\), \\(s\\), and  for \\(p = 0\\) 
$$o = \lfloor \dfrac{i - k}{s} \rfloor + 1$$

* Figure 1.11  provides an example for \\(i = 5\\), \\(k = 3\\), \\(p = 0\\) and \\(s = 2\\),
    therefore output size \\(o = \lfloor \dfrac{5 - 3}{2} \rfloor + 1 = 2\\).

* **_NOTE_** : The floor function in relationship-5 accounts for the fact that sometimes input size is such that kernel 
would not be able to reach all the input units.

* Figure 1.12 illustrates this
{% include image.html url="../assets/arbitrary_padding_strides.PNG" description="Figure 1.12 Arbitrary padding and Strides" %}

<a name="L46"></a>
### Zero padding, non-unit strides
This is the more general case, convolving over a zero padded input using a non-unit strides. 
We can derive by applying relationship-5 on effective input size of \\(i + 2p\\)

**Relationship-6**: For any \\(i\\), \\(k\\), \\(s\\), and \\(p\\) 
$$o = \lfloor \dfrac{i + 2p - k}{s} \rfloor + 1$$
{% include image.html url="../assets/arbitrary_padding_strides-1.PNG" description="Figure 1.13 Arbitrary padding and Strides" %}

* Figure 1.13  provides an example for \\(i = 5\\), \\(k = 3\\), \\(s = 2\\) and \\(p = 1\\),
    therefore output size \\(o = \lfloor \dfrac{5 + 2*1 - 3}{2} \rfloor + 1 = 5\\).
* Figure 1.12  provides an example for \\(i = 6\\), \\(k = 3\\), \\(s = 2\\) and \\(p = 1\\),
    therefore output size \\(o = \lfloor \dfrac{6 + 2*1 - 3}{2} \rfloor + 1 = 5\\).

Observe that even though both has different size inputs \\(i = 5\\) and \\(i = 6\\), output size after convolution is same \\(o = 5\\) for both.
As discussed before, this is due to the fact that kernel is not able to reach all the input units. 


_continued in [part 2](https://nrupatunga.github.io/convolution-2/)_

<a name="LR"></a>
## References
[1] [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)
