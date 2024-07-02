# Computer vision (CV) - Notes

## Table of Contents

- [What's Computer Vision (CV)?](#whats-computer-vision-cv)
- [How It Works?](#how-it-works)
- [Computer Vision Pipeline Architecture](#computer-vision-pipeline-architecture)
- [Applications](#applications)
- [Tools \& Frameworks](#tools--frameworks)
- [Computer Vision Sub-fields/domains](#computer-vision-sub-fieldsdomains)
- [CV Algorithms \& Applications](#cv-algorithms--applications)
- [Computer Vision \& Deep Learning \& LLMs \& GenAI](#computer-vision--deep-learning--llms--genai)
- [CV Datasets - Non-exhautive list](#cv-datasets---non-exhautive-list)
- [Hello World!](#hello-world)
- [CV Projects](#cv-projects)
- [Top Computer Vision Conferences](#top-computer-vision-conferences)
- [Computer Vision Researcher - Starting Resources](#computer-vision-researcher---starting-resources)
- [References](#references)
 

### What's Computer Vision (CV)?

CV uses artificial intelligence (AI) to interpret visual data (mainly digital images and videos ...) to take actions and make recommendations.

### How It Works? 

If [AI](../ai/ai-notes.md) enables computers to think, **computer vision** enables them to see, observe and understand the world.

CV uses a variety of algorithms and a special [deep learning](../ai/deep-learning-notes/dl-notes.md) neural nets called [Convolutional Neural Networks (CNN)](../ai/deep-learning-notes/neural-nets/convolutional-neural-network.ipynb).

## Computer Vision (Basic) Pipeline Architecture

```mermaid
graph LR
    A[Image Acquisition] ==> B[Image Processing]
    B ==> C[Feature Extraction]
    C ==> D[Output, Interpretation & Analysis]

    style A fill:#EEE,stroke:#333,stroke-width:4px
    style B fill:#F88,stroke:#333,stroke-width:4px
    style C fill:#4F4,stroke:#333,stroke-width:4px
    style D fill:#33F,stroke:#333,stroke-width:4px
```

## Applications 

- Face recognition
- Optical Character Recognition (OCR)
- Content-based image retrieval
- Autonomous vehicles & Robotics
- Augmented reality (AR)
- Remote sensing
...

## Tools & Frameworks

List of the Most Popular Computer Vision Tools in 2022 - by Viso.ai: 
1. OpenCV
2. Viso Suite
3. Pytorch
4. TensorFlow
5. CUDA
6. MATLAB
7. Keras
8. SimpleCV
9. BoofCV
10. [CAFFE](https://caffe.berkeleyvision.org/)
11. OpenVINO
12. DeepFace
...


## Computer Vision Sub-fields/domains

![cv-diagram](./docs/Computer-Vision-aplications-and-research-fields.png)

Src: [Rui Nobrega](https://www.researchgate.net/figure/Computer-Vision-aplications-and-research-fields_fig1_258013244)

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

**Structure from motion algorithms**
- Can reconstruct a sparse 3D point model of a large complex scene from hundreds of partially overlapping photographs

**Stereo matching algorithms**
- Can build a detailed 3D model of a building façade from hundreds of differently exposed photographs taken from the Internet

**Person tracking algorithms**
- Can track a person walking in front of a cluttered background

**Face detection algorithms**
- Coupled with color-based clothing and hair detection algorithms, can locate and recognize the individuals in this image

## Computer Vision & Deep Learning

**Non-exhautive list of Conv Nets (CNN) Archictectures**

- [AlexNet](https://en.wikipedia.org/wiki/AlexNet), BN-AlexNet, BN-INI, ENet, GoogLeNet (Inception-v1), ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152 , Inception-v3 , Inception-v4, VGG-16, VGG-19, ViT, ...

![](https://camo.githubusercontent.com/728e3980abb477ced930abd05634becfcc9c74c3369a713304e39f4baaad3b43/68747470733a2f2f6d69726f2e6d656469756d2e636f6d2f76322f726573697a653a6669743a313430302f666f726d61743a776562702f312a6e31366c6a336c536b7a326d694d635f3563766b72412e6a706567)
(From: [Benchmark Analysis of Representative Deep Neural Network Architectures - Luigi Celona (@CeLuigi) - 2018](https://arxiv.org/pdf/1810.00736.pdf))

**Deep Learning Research Notes**

- [Deep Learning Neural Nets](..\ai\deep-learning-notes\neural-nets)
- [Deep Learning Neural Nets Architectures](..\ai\deep-learning-notes\neural-nets-architecture-notes.md)
- [Convolutional Neural Network (CNN) - Notebook](https://github.com/afondiel/computer-science-notes/blob/master/ai/deep-learning-notes/neural-nets/notebook/convolutional-neural-network.ipynb)
- [The State of Computer Vision at Hugging Face](https://huggingface.co/blog/cv_state)
- [Computer Vision & Keras resources](https://keras.io/examples/vision/)

## Computer Vision & Generative AI (LLMs, VLMs, Vision-LLMs, ...)

**Vision GenAI**
- [Generative Adversarial Network (GAN)](https://en.wikipedia.org/wiki/Generative_adversarial_network)
- [OpenAI DALL-E](https://en.wikipedia.org/wiki/DALL-E)
- [Midjourney](https://en.wikipedia.org/wiki/Midjourney)
- [Stable Diffusion](https://en.wikipedia.org/wiki/Stable_Diffusion)

**Vision-LLMs**
- [LLM-in-Vision - Yong-Lu Li(@DirtyHarryLYL)](https://github.com/DirtyHarryLYL/LLM-in-Vision)
- [Computer vision and Large Language Models (LLM)](https://www.moroccan-in-asia.com/post/computer-vision-and-large-language-models-llm)
- [A Dive into Vision-Language Models - HuggingFace](https://huggingface.co/blog/vision_language_pretraining)
- [How do Large Language Models (LLMs) Integrate with Computer Vision?](https://www.chooch.com/blog/how-to-integrate-large-language-models-with-computer-vision/)
- [How do LLMs work with Vision AI? | OCR, Image & Video Analysis](https://techcommunity.microsoft.com/t5/microsoft-mechanics-blog/how-do-llms-work-with-vision-ai-ocr-image-amp-video-analysis/ba-p/3835661)
- [It's not just words: LLMs in Computer Vision - Microsoft Reactor](https://www.youtube.com/watch?v=ss-kyogPRNo)

**Vision-Language Models**
- [A Dive into Vision-Language Models - HF](https://huggingface.co/blog/vision_language_pretraining) 
- [Vision Language Models Explained - HF](https://huggingface.co/blog/vlms)


**Computer Vision Paper Review**

- [paper-review-and-code](./research-papers/paper-review-and-code/)

## Computer Vision & Multimodality

- [Multimodal Models and Computer Vision: A Deep Dive - Roboflow](https://blog.roboflow.com/multimodal-models/)
- [Multimodal Deep Learning: Definition, Examples, Applications - v7](https://www.v7labs.com/blog/multimodal-deep-learning-guide)
- [Understanding the Role of Multimodal Models in Computer Vision - Inveniam](https://www.inveniam.fr/understanding-the-role-of-multimodal-models-in-computer-vision)
- [Multimodal embeddings - Microsoft](https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/concept-image-retrieval)
- [V-JEPA: The next step toward Yann LeCun’s vision of advanced machine intelligence (AMI)](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)
- [What is Meta’s new V-JEPA model? [Explained] - GFG](https://www.geeksforgeeks.org/what-is-metas-new-v-jepa-model-explained/)
- [How to use I-JEPA for image classficiation](https://discuss.huggingface.co/t/how-to-use-i-jepa-for-image-classficiation/46934)
- [vit-large-patch16-v-jepa - @elsr](https://huggingface.co/nielsr/vit-large-patch16-v-jepa)

## CV Datasets (Non-exhautive list)

1. [ImageNet](https://www.image-net.org/index.php)
2. [COCO (Common Objects in Context)](https://cocodataset.org/#home)
3. Open Images
4. Pascal VOC (Visual Object Classes)
5. Cityscapes
6. SUN (Scene Understanding)
7. LFW (Labeled Faces in the Wild)
8. CelebA
9. CIFAR-10 and CIFAR-100
10. MNIST

Src: [Top 10 Open-Source Datasets for Computer Vision in 2023](https://www.analyticsinsight.net/top-10-open-source-datasets-for-computer-vision-in-2023/)

Automotive dataset: 

- [Top 7 Automotive Datasets for in Computer Vision Projects ](https://www.labellerr.com/blog/top-7-automotive-datasets-for-computer-vision-projects/)
- [Stanford Cars Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)
- [Top Self Driving Datasets](https://universe.roboflow.com/browse/self-driving)

[SOTA Computer Vision Dataset - @paperwithcode](https://paperswithcode.com/area/computer-vision)

## Hello World!

- Check out the [Computer Vision Challenge](https://github.com/afondiel/computer-vision-challenge) `L0 & L1` for a collection of fundational CV projects.

## CV Projects

Please refer to the [lab](./lab/) repository. 

## Computer Vision Project in Industry

- [Meta Research Computer Vision Projects](https://research.facebook.com/publications/research-area/computer-vision/)
- [Google Vision AI Framework](https://cloud.google.com/vision?hl=en)
- [NVIDIA Computer Vision Products and Services](https://www.nvidia.com/en-us/glossary/computer-vision/)
- [Amazon & AWS Computer Vision Frameworks](https://aws.amazon.com/computer-vision/)
- [Apple vision Frameworks](https://developer.apple.com/documentation/vision/)
- [OpenAI Vision Tools & Framework](https://platform.openai.com/docs/guides/vision)
- [The State of Computer Vision at Hugging Face](https://huggingface.co/blog/cv_state)
- [Langchain & Computer Vision Tooling](https://www.toolify.ai/gpts/mastering-computer-vision-with-langchain-custom-tools-359661)


## Top Computer Vision Conferences

- Conference on Neural Information Processing Systems (NeurIPS)
- International Conference on Learning Representations (ICLR)
- Computer Vision and Pattern Recognition (CVPR)
- European Conference on Computer Vision (ECCV)
- International Conference on Computer Vision (ICCV)
- Conference on Robot Learning (CoRL)
- British Machine Vision Conference (BMVC)
- ...

Src: [The Ultimate Guide to Top Computer Vision Conferences: Where Tech Visionaries Unite](https://augmentedstartups.medium.com/the-ultimate-guide-to-top-computer-vision-conferences-where-tech-visionaries-unite-72f99c990ac1)

## Computer Vision Researcher - Starting Resources   

- [Quora - great feed](https://www.quora.com/How-do-I-start-research-in-computer-vision)
- [Reddit - great feed](https://www.reddit.com/r/computervision/comments/13s9vm7/how_to_start_research/)
- [International Journal of Computer Vision](https://www.springer.com/journal/11263/)
- [Must read CV papers](https://www.quora.com/What-are-the-best-papers-about-computer-vision#NyMbP)
- [Vision and Learning Lab - Old Website](http://vision.stanford.edu/)
- [Vision and Learning Lab (SVL) - New Website](https://svl.stanford.edu/)
- [Car dataset in multiple scenes for AI & Computer Vision](https://aws.amazon.com/marketplace/pp/prodview-6i7gc6mi7hlgs#offers)


## References

**Computer Vision Roadmap**:
- [Introduction to Computer Vision / Roadmap to CV Developer in 2022](https://www.karachidotai.com/introduction-to-computer-vision)
- [Computer Vision Tutorial for Beginners | Learn Computer Vision](https://www.projectpro.io/data-science-in-python-tutorial/computer-vision-tutorial-for-beginners)
- https://www.youtube.com/watch?v=jLcuVu5xdDo 
- https://www.youtube.com/watch?v=DfPAvepK5Ws

**Wikipedia**: 

- Essential notes

  - [Computer Vision](https://en.wikipedia.org/wiki/Computer_vision)
  - [Machine Perception](https://en.wikipedia.org/wiki/Machine_perception)
  - [Depth Perception](https://en.wikipedia.org/wiki/Depth_perception)
  - [Understanding Depth Perception in Computer Vision - Nasscom](https://coe-dsai.nasscom.in/understanding-depth-perception-in-computer-vision/)

- Environment
  - [Light](https://en.wikipedia.org/wiki/Light)
  - [Electromagnetic Radiation](https://en.wikipedia.org/wiki/Electromagnetic_radiation)
  - [Electromagnetic Spectrum](https://en.wikipedia.org/wiki/Electromagnetic_spectrum)
  - [Optics](https://en.wikipedia.org/wiki/Optics)
  - [Color Theory](https://en.wikipedia.org/wiki/Color_theory)

- Human perception 
  - [Human Eye](https://en.wikipedia.org/wiki/Human_eye)
  - [Visual System](https://en.wikipedia.org/wiki/Visual_system)
  - [Visual Cortex](https://en.wikipedia.org/wiki/Visual_cortex)

- Signal Processing

  - [Multidimensional Signal Processing](https://en.wikipedia.org/wiki/Category:Multidimensional_signal_processing)
  - [Image Processing](https://en.wikipedia.org/wiki/Category:Image_processing)
  - [Video Processing](https://en.wikipedia.org/wiki/Category:Video_processing)
  - [Geometry Processing](https://en.wikipedia.org/wiki/Category:Geometry_processing)
  - [Content Based Image Retrieval - (CBIR)](https://en.wikipedia.org/wiki/Content-based_image_retrieval)

- Geometry
  - [Stereometry or Solid Geometry - 3D (Objects) Geometry](https://en.wikipedia.org/wiki/Solid_geometry)
  - [Geometry Processing](https://en.wikipedia.org/wiki/Geometry_processing)
  - [3D space](https://en.wikipedia.org/wiki/Three-dimensional_space)

- Deep Learning

  - [Convolutional Neural Network - CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)

**Vision sensors**
- Camera Sensor: 
  - [How does a camera work? - Branch Education ](https://www.youtube.com/watch?v=B7Dopv6kzJA)
  - [The Science of Camera Sensors - Filmmaker IQ](https://www.youtube.com/watch?v=MytCfECfqWc)
  - [Understanding The Digital Image Sensor](https://thinklucid.com/tech-briefs/understanding-digital-image-sensors/)
  - [Image Sensor](https://en.wikipedia.org/wiki/Image_sensor)
  - [The ultimate image sensor guide: EMCCD vs. SCMOS vs. CMOS vs. CCD](https://www.e-consystems.com/blog/camera/technology/the-ultimate-image-sensor-guide-emccd-vs-scmos-vs-cmos-vs-ccd/)
  - [Types Of Camera Sensor](https://www.photometrics.com/learn/camera-basics/types-of-camera-sensor)
- Optical sensors:
  - [Electro-optical sensor](https://en.wikipedia.org/wiki/Electro-optical_sensor)
  - [Optical Sensor Basics and Applications](https://www.elprocus.com/optical-sensors-types-basics-and-applications/)


**Machine Vision**
- [Machine Vision](https://en.wikipedia.org/wiki/Machine_vision)
- [A Practical Guide to Machine Vision Lighting](https://www.ni.com/en/shop/choosing-the-right-hardware-for-your-vision-applications/a-practical-guide-to-machine-vision-lighting.html#:~:text=Back%20to%20top-,Vision%20Illumination%20Sources%20and%20Spectral%20Content,(mercury)%2C%20and%20xenon.)

**Lectures & Tutorials**:

- Stanford
  - [CS131 Computer Vision: Foundations and Applications - Stanford - Fall 2017](http://vision.stanford.edu/teaching/cs131_fall1718/)
  - [CS231A: Computer Vision, From 3D Reconstruction to Recognition Winter 2023](https://web.stanford.edu/class/cs231a/course_notes.html) 
  - [CS231n Computer Vision Stanford - 2017](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)
  - [Deep Learning for Computer Vision (Andrej Karpathy, OpenAI)](https://www.youtube.com/watch?v=u6aEYuemt0M)

- Udemy Course:
  - [The Complete Self-Driving Car Course Applied Deep-Learning - Udemy - Notes](https://github.com/afondiel/The-Complete-Self-Driving-Car-Course-Udemy/blob/main/self-driving-cars-dl-notes.md)

- FreeCodeCamp:
  - [OpenCV Course - Full Tutorial with Python - 4H - FreeCodeCamp](https://www.youtube.com/watch?v=oXlwWbU8l2o&t=69s)
  - [Deep Learning for Computer Vision with Python and TensorFlow – Complete Course](https://www.youtube.com/watch?v=IA3WxTTPXqQ)
  - [OpenGL Course - Create 3D and 2D Graphics With C++](https://www.youtube.com/watch?v=45MIykWJ-C4)
- Torchvision: https://www.youtube.com/watch?v=CU6bTEClzlw

- University of Toronto:
  - [CSC2541:Visual Perception for Autonomous Driving - Winter 2016](http://www.cs.toronto.edu/~urtasun/courses/CSC2541/CSC2541_Winter16.html)

- [IBM - What is computer vision?](https://www.ibm.com/hk-en/topics/computer-vision#:~:text=Computer%20vision%20is%20a%20field,recommendations%20based%20on%20that%20information.)
- [Computer Vision Tutorial - GeeksForGeeks](https://www.geeksforgeeks.org/computer-vision/)
- [v7labs - 27+ Most Popular Computer Vision Applications and Use Cases in 2022](https://www.v7labs.com/blog/computer-vision-applications)
- [viso.ai - The 12 Most Popular Computer Vision Tools in 2022](https://viso.ai/computer-vision/the-most-popular-computer-vision-tools/)
- [Lines Detection with Hough Transform](https://towardsdatascience.com/lines-detection-with-hough-transform-84020b3b1549)
- [What are the different subfields in computer vision? - Quora](https://www.quora.com/What-are-the-different-subfields-in-computer-vision)

**Computer Vision & Transportation**

- **Tesla Vision vs Lidar**: 
  - [Tesla autonomy neural networks How AI neural networks function in Tesla - By Andrej Karpathy](https://www.youtube.com/watch?v=eZOHA6Uy52k)
  - [Tesla Vision vs LIDAR](https://www.youtube.com/watch?v=W-ubNvS0RGU)
  - [How Tesla Is Using Artificial Intelligence to Create The Autonomous Cars Of The Future](https://bernardmarr.com/how-tesla-is-using-artificial-intelligence-to-create-the-autonomous-cars-of-the-future/)
  - [LiDAR vs Computer Vision: Does Waymo Have A Better Strategy Than Tesla? - Joe Scott](https://www.youtube.com/watch?v=HmjuEikY4ew)
  - [Tesla Vision vs UltraSonics Sensors (USS)](https://www.youtube.com/watch?v=1imyiPbYB24)

- **Tesla AI Day**: 
  - [Tesla Autonomy Day - 2019](https://www.youtube.com/watch?v=Ucp0TTmvqOE)
  - [Tesla Battery Day - 2020 - during covid](https://www.youtube.com/watch?v=l6T9xIeZTds)
  - [Tesla AI Day - 2021](https://www.youtube.com/watch?v=j0z4FweCy4M&t=37s)
  - [Tesla AI Day - 2022](https://www.youtube.com/watch?v=ODSJsviD_SU&t=3480s)

**Computer Vision & Space Exploration**
- [Computer Vision & Space Explorations - Notes](./lab/cv-space-explorations.md)
- [Space Imagery Data - Nasa earthdata](https://www.earthdata.nasa.gov/)
- [Computer Vision and Remote Sensing](https://www.satimagingcorp.com/applications/artificial-intelligence-ai/computer-vision/)
- [Computer Vision in Google Maps](https://blog.google/products/maps/google-maps-immersive-view-routes/#:~:text=Using%20machine%20learning%20to%20extract,to%20understand%20what's%20in%20them.)
- [Beyond Human Sight: How AI Computer Vision Reimagines Space Exploration.](https://devisionx.com/how-ai-computer-vision-reimagines-space-exploration/)
- [Sophisticated Satellite Vision ](https://viso.ai/computer-vision/computer-vision-trends/)

**Research Notes resources**:
- [Computer Science Notes - cv resources](https://github.com/afondiel/computer-science-notes/blob/master/computer-vision-notes/cv-notes.md)
- [CONVOLUTIONAL NEURAL NETWORK (CNN) - Notes](https://github.com/afondiel/computer-science-notes/blob/master/ai/ml-notes/deep-learning-notes/neural-nets-architecture-notes.md)
- [Convolutional Neural Network(CNN) - notebook](https://github.com/afondiel/computer-science-notes/blob/master/ai/deep-learning-notes/neural-nets/convolutional-neural-network.ipynb)

**Papers**
- [Research Papers](https://github.com/afondiel/computer-science-notes/tree/master/computer-vision-notes/research-papers)
- [Generative Adversarial Networks - Gan](https://arxiv.org/abs/1406.2661) 
- [OpenAI Dall-E](https://openai.com/blog/dall-e-introducing-outpainting/)

**Books** 
- [Computer Vision - Resources](https://github.com/afondiel/cs-books/tree/main/computer-vision)
 

> ## "Vision is a picture of the future that produces passion" ~ Bill Hybels

