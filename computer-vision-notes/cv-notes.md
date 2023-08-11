# Computer vision (CV) - Notes

## Table of Contents

- [What is computer vision?](#what-is-computer-vision)
- [Applications](#applications)
- [Tools/frameworks](#toolsframeworks)
- [Image processing notes](#image-processing-notes)
  - [useful concepts](#useful-concepts)
  - [Operations](#operations)
- [Finding Lanes application](#finding-lanes-application)
- [References :](#references-)

## Overview

This is a Computer Vision "Hello World" resources.
## What is computer vision?

CV uses artificial intelligence (AI) to interpret visual data (digital images and videos ...) to take actions and make recommendations.

- If AI enables computers to think, computer vision **enables them to see, observe and understand.**

## Applications 

- Augmented reality (AR)
- Autonomous vehicles
- Content-based image retrieval
- Face recognition
- Optical character recognition
- Remote sensing
- Robotics
...

## Tools/frameworks
List of the Most Popular Computer Vision Tools in 2022 - by Viso.ai : 
1. OpenCV
2. Viso Suite
3. Pytorch
4. TensorFlow
5. CUDA
6. MATLAB
7. Keras
8. SimpleCV
9. BoofCV
10. CAFFE
11. OpenVINO
12. DeepFace
...

## Computer Vision Sub-fields/domains

![cv-diagram](./documentation/cv-diagram.png)

Sub-domains of computer vision include :  
- scene reconstruction, 
- object detection, 
- event detection, 
- activity recognition, 
- video tracking, 
- object recognition, 
- 3D pose estimation, 
- learning, indexing, 
- motion estimation, 
- visual servoing, 
- 3D scene modeling, 
- image restoration
- ...
    

## CV Algorithms & Applications

**Finding Lanes application**

- Setup the environment : 
1. load image => OK 
    => pip install opencv-python

2. Gray Scale (Niveau de Gris (NG))

Detection process

- goal : (original img) => gradient img 

3. original img (RGN : 3 channels ) => grayscale img (1 channel)
4. Use Hough Transform : a polar parametric space (parameters of the line)
    - pixels can mathematically be represented in a scatter graph 
    - we can the draw a (straight) line => (y =  mx + b )
    - convert cartesian plan (m, b) to polar plan (Theta, Rho)
    - the (new)pixel is the intersection of the lines in Hough space
    - Also the (new) pixel represents the line chosen by the system of votes (numbers of intersections ) of severals courves of the plan 
    - vote : number of intersection  
5. Optimization : 

**Structure from motion algorithms**
- Can reconstruct a sparse 3D point model of a large complex scene from hundreds of partially overlapping photographs

**Stereo matching algorithms**
- Can build a detailed 3D model of a building façade from hundreds of differently exposed photographs taken from the Internet

**Person tracking algorithms**
- Can track a person walking in front of a cluttered background

**Face detection algorithms**
- Coupled with color-based clothing and hair detection algorithms, can locate and recognize the individuals in this image

## Computer Vision and Deep Learning

- [Deep Learning NN](..\ai\ml-notes\deep-learning-notes\neural-nets)
- [Deep Learning NN Architectures](..\ai\ml-notes\deep-learning-notes\neural-nets-architecture-notes.md)

- [Convolutional Neural Network(CNN) - notebook](https://github.com/afondiel/research-notes/blob/master/ai/deep-learning-notes/neural-nets/convolutional-neural-network.ipynb)

## CV & AI Models - Non-exhautive list

- AlexNet
- BN-AlexNet
- BN-INI
- ENet
- GooLeNet
- ResNet-18
- ResNet-34
- ResNet-50
- ResNet-101
- ResNet-152
- Inception-v3 
- Inception-v4
- VGG-16
- VGG-19

## CV & Dataset - Non-exhautive list

1. ImageNet
2. COCO (Common Objects in Context)
3. Open Images
4. Pascal VOC (Visual Object Classes)
5. Cityscapes
6. SUN (Scene Understanding)
7. LFW (Labeled Faces in the Wild)
8. CelebA
9. CIFAR-10 and CIFAR-100
10. MNIST
    
Src: [Top 10 Open-Source Datasets for Computer Vision in 2023](https://www.analyticsinsight.net/top-10-open-source-datasets-for-computer-vision-in-2023/)

## References

- Wikipedia - Quick notes: 
  - [Computer Vision](https://en.wikipedia.org/wiki/Computer_vision)
  - [Human Visual System](https://en.wikipedia.org/wiki/Visual_system)
  - [Optics](https://en.wikipedia.org/wiki/Optics)
  - [Light](https://en.wikipedia.org/wiki/Light)
  - [Electromagnetic Spectrum](https://en.wikipedia.org/wiki/Electromagnetic_spectrum)
  - [Machine Perception](https://en.wikipedia.org/wiki/Machine_perception)
  - [Convolutional Neural Network - CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- Quora
  - [What are the different subfields in computer vision?](https://www.quora.com/What-are-the-different-subfields-in-computer-vision)

- Books 
  - [Computer Vision - Resources](https://github.com/afondiel/cs-books/tree/main/computer-vision)

- Courses - Blog posts
  - **Computer Vision roadmap** :
    - [Introduction to Computer Vision / Roadmap to CV Developer in 2022](https://www.karachidotai.com/introduction-to-computer-vision)
    - [Computer Vision Tutorial for Beginners | Learn Computer Vision](https://www.projectpro.io/data-science-in-python-tutorial/computer-vision-tutorial-for-beginners)
    - https://www.youtube.com/watch?v=jLcuVu5xdDo 
    - https://www.youtube.com/watch?v=DfPAvepK5Ws

  - [Computer Vision Tutorial - GeeksForGeeks](https://www.geeksforgeeks.org/computer-vision/)
  - [CS131 Computer Vision: Foundations and Applications - Stanford](http://vision.stanford.edu/teaching/cs131_fall1718/)
  - [IBM - What is computer vision?](https://www.ibm.com/hk-en/topics/computer-vision#:~:text=Computer%20vision%20is%20a%20field,recommendations%20based%20on%20that%20information.)
  - [v7labs - 27+ Most Popular Computer Vision Applications and Use Cases in 2022](https://www.v7labs.com/blog/computer-vision-applications)
  - [viso.ai - The 12 Most Popular Computer Vision Tools in 2022](https://viso.ai/computer-vision/the-most-popular-computer-vision-tools/)
  - [Lines Detection with Hough Transform](https://towardsdatascience.com/lines-detection-with-hough-transform-84020b3b1549)

- YT Videos & Free Tutorials: 
  - [Deep Learning for Computer Vision (Andrej Karpathy, OpenAI)](https://www.youtube.com/watch?v=u6aEYuemt0M)
  - [OpenCV Course - Full Tutorial with Python - 4H - FreeCodeCamp](https://www.youtube.com/watch?v=oXlwWbU8l2o&t=69s)
  - [Deep Learning for Computer Vision with Python and TensorFlow – Complete Course](https://www.youtube.com/watch?v=IA3WxTTPXqQ)
  - [OpenGL Course - Create 3D and 2D Graphics With C++](https://www.youtube.com/watch?v=45MIykWJ-C4)
  - Torchvision: https://www.youtube.com/watch?v=CU6bTEClzlw
  - 

- Notebooks
  - [Convolutional Neural Network(CNN) - notebook](https://github.com/afondiel/research-notes/blob/master/ai/deep-learning-notes/neural-nets/convolutional-neural-network.ipynb)

- Papers 
  - [Research Papers Archives](https://github.com/afondiel/research-notes/tree/master/computer-vision-notes/research-papers)
  - [Generative Adversarial Networks - Gan](https://arxiv.org/abs/1406.2661) 
  - [OpenAI Dall-E](https://openai.com/blog/dall-e-introducing-outpainting/)
   

> ## "Vision is a picture of the future that produces passion" ~ Bill Hybels

