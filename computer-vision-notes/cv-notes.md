# Computer vision (CV) - Notes

## Agenda

  - [Intro](#intro)
  - [Applications](#applications)
  - [Tools/frameworks](#toolsframeworks)
  - [Image processing notes:](#image-processing-notes)
    - [useful concepts](#useful-concepts)
    - [Operations](#operations)
  - [Finding Lanes application  :](#finding-lanes-application--)
- [References :](#references-)


## Intro
- Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.

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
3. TensorFlow
4. CUDA
5. MATLAB
6. Keras
7. SimpleCV
8. BoofCV
9. CAFFE
10. OpenVINO
11. DeepFace
...

## Image processing notes

### Useful concepts

- image : a 2D matrix (n x m , array of pixels) 
- types of images : 
  - RGB(Red Green Blue - original colored image) : 3D (dim (W x H x L)  => 3 channels
  - GrayScale (Gradient) : 2D (dim(W x H)) => 1 channel
- PIXEL (Picture ELement): it is the smallest element of a digital image (2D)
  - light intensity of some location in the img 
- VIXEL (Volumic PIXEL): pixel of a 3D image
  
GrayScale image (GS) : is one in which each pixel is a single sample representing only an amount of light. GS are composed exclusively of shades of gray, with the contrast ranging from black at the weakest intensity to white at the strongest
.
- Edge : rapid change in brightness
- Edge Detection : identifying sharp changes in intensity in adjacent(neighboring) pixels 
  
            ^
            |
            +--+-----+---+-----+--+
            |0 | 255 | 0 | 255 | 0|
            +--+-----+---+-----+--+
            + : represents the edge 
pixel representations

        [0 0 255 255]
        [0 0 255 255]
        [0 0 255 255]
        [0 0 255 255]
        [0 0 255 255]
        0 ---------> 255
        where : 
            0 : Black 
            255 : White 

- Gradient : measure of change in brightness over adjacent pixels 
  - strong gradient (0 - 255)
  - small gradient (0 - 15 ...)

- Hystogram: it is the distribution (frequency) of grayscale (GS) of an image

        255
        ^(Pixels)
        |                   +
        |                 + || +
        |     +          +  ||  +
        |   + || +      +   ||   + (density/frequency?)
        |  +  ||   + +      ||
        |     ||     ||     ||
        |     ||     ||     ||           
        +-----+------+-------+---------> (GS : samples)
        0                              255


### Operations
- **Region of ​​interest(ROI)**: a portion of an image that you want to filter or perform some other operation on
- **Segmentation**: operation which consists of extracting geometric primitives from an image.
    - The most used primitives:
        - Segments (outlines)
        - Surfaces( regions)
        - classification or thresholding
- **Calibration**: operation which consists in finding the relation between the coordinates in the image and the coordinates in the space of the scene.
- **Filter**: takes an input image and produces a (cleaner) output image
- **LUT(Lool Up Table)**: conversion table that groups neighboring NG pixels into a single value.
    - Matrix with different pixel values
    - By doing operations on the LUTs it allows to process the image
- **Recale**: geometric transformation technique allowing to go from a source image to a target image (destination)

- **Morphological**:
  - Skeleton
  - Erosion
  - Dilation
- **Frequency**
  - Frame
  - Equalize
  - Opening(object)
  - Closure(hole)
  - Filter(freq)

- **Operators**:
  - img(src) -> Img(destination)
  - img (src) -> information set
  - information set -> img(destination)
    


## Finding Lanes application
Setup the environment : 
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

# References
- Wikipedia - Quick notes: 
  - [Computer Vision](https://en.wikipedia.org/wiki/Computer_vision)
  - [Human Visual System](https://en.wikipedia.org/wiki/Visual_system)
  - [Optics](https://en.wikipedia.org/wiki/Optics)
  - [Light](https://en.wikipedia.org/wiki/Light)
  - [Electromagnetic Spectrum](https://en.wikipedia.org/wiki/Electromagnetic_spectrum)
  - [Machine Perception](https://en.wikipedia.org/wiki/Machine_perception)
  - [Convolutional Neural Network - CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)

- Books 
  - [Computer Vision - resources](https://github.com/afondiel/cs-books/tree/main/computer-vision)
- Courses - blog posts
  - [CS131 Computer Vision: Foundations and Applications - Stanford](http://vision.stanford.edu/teaching/cs131_fall1718/)
  - [v7labs - 27+ Most Popular Computer Vision Applications and Use Cases in 2022](https://www.v7labs.com/blog/computer-vision-applications)
  - [viso.ai - The 12 Most Popular Computer Vision Tools in 2022](https://viso.ai/computer-vision/the-most-popular-computer-vision-tools/)
  - [Lines Detection with Hough Transform](https://towardsdatascience.com/lines-detection-with-hough-transform-84020b3b1549)
  - [OpenGL Course - Create 3D and 2D Graphics With C++](https://www.youtube.com/watch?v=45MIykWJ-C4)

- notebooks
  - [Convolutional Neural Network(CNN) - notebook](https://github.com/afondiel/research-notes/blob/master/ai/deep-learning-notes/neural-nets/convolutional-neural-network.ipynb)

- papers 
  - [research-papers archives](https://github.com/afondiel/research-notes/tree/master/computer-vision-notes/research-papers)
  - [Generative Adversarial Networks - Gan](https://arxiv.org/abs/1406.2661) 
  - [OpenAI Dall-E](https://openai.com/blog/dall-e-introducing-outpainting/)
   

> ## "Vision is a picture of the future that produces passion" ~ Bill Hybels
