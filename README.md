# Plant-Disease-Detection
## Table of Contents

    1    Project Title    
    2    Domain    
    3    Problem Statement    
    4    Project Description    
    5    Assumptions
    8    Technologies to be used      
    11   Tools    
    12   Advantages of this Project    
    13   Future Scope and further enhancement of the Project  
    15   Conclusion    
    16   References    
    
 ## Project Title
In Agriculture, leaf diseases have grown to be a dilemma as it can cause a significant diminution in both the quality and quantity of agricultural yields. Thus, automated recognition of diseases on leaves plays a crucial role in the agriculture sector. This Crop Disease detection project is based on the same idea which can be used to detect disease in specific crops on which it is trained.

## Domain
‘Agriculture is the backbone of Indian Economy’, said by Mahatma Gandhi. Even today, the situation is still the same, with almost the entire economy being sustained by agriculture. It contributes 16% of the overall GDP and accounts for employment of approximately 52% of the Indian population. 

## Problem Statement
In Agriculture sector plants or crops, cultivation has seen fast development in both the quality and quantity of food production. However, the presence of diseases on crops especially on leaves has hindered the quality of agricultural goods. This severe effect can disturb any nation's economy especially of those where 70% of the inhabitants rely on the products from the agricultural sector for their livelihood and endurance. This problem can be solved by the detection of crop diseases and to detect crop disease this project can be used. This project will diagnose the disease based on images of leaves.

## Project Description
Functional requirements define the internal workings of the software: that is, the technical details, data manipulation and processing and other specific functionality that show how the use cases are to be satisfied. They are supported by non-functional requirements, which impose constraints on the design or implementation.
IMAGE PROCESSING Digital image processing is the use of computer algorithms to perform image processing on digital images. It allows a much wider range of algorithms to be applied to the input data and can avoid problems such as the build-up of noise and signal distortion during processing. Since images are defined over two dimensions (perhaps more) digital image processing may be model in the form of [multidimensional systems].
The following steps are followed for detecting disease in crop:

### Image Acquisition-: 
The real-time images are fed directly from the camera. For further analysis, proper visibility and easy analysis of images, white background is created because most of the leaves color vary from red to green for exact segmentation.

### Image Pre-processing-: 
Image preprocessing is required to resize captured image from high resolution to low resolution. The image resizing can be done through the process of interpolation. 
Batch Normalization is also used normalize data.

### Feature Extraction and Feature Selection-: 
Feature Extraction is one of the most interesting steps of image processing to reduce the efficient part of an image or dimensionality reduction of interesting parts of an image as a compact feature vector. Feature reduction representation is useful when the image size is large and required to rapidly complete tasks such as image matching and retrieval. This part is handled By our CNN Layer which is used for feature detection and for selection we use Max Pooling and dropout Layer.

### Model Training-:
It is done after model curation which is done on the basis of all above information since we are using deep learning most of tha task or almost all tasks are handled by our model itself so to perform all above operation we define layers in our model. After curation of Model we train our model by setting our Hyper-perameters. After training part is performed our model is ready for the process of Image Classification.

### Image Classification-: 
It consists of model that contains pre-defined patterns that are compared with detected objects to classify them in a proper category. Classification will be executed on the basis of spectral defined feature such as density, texture etc. Image classification is performed using Convolution Neural Networks and Deep Learning, and it has introduced the Convolution Neural Network (CNN) as a new area in machine learning and is applied to detect crop disease through classification of image.

### Web App Creation Using Flask-:
Simple web app is created using flask which is a micro-framework for web app creation using python. For this one python file is created which is connected with multiple Javascript, HTML and CSS since I have created a small web app rather than a fancy and complex web app because our aim is just to present our model functionality so I have used one file of each type rather than multiple files. Since flask app can run on local system but to share the working of model globally I have deployed Model on Heroku server.

### Model Deployment-:

Our Model is deployed on Heroku Server which is easy to integrate by connecting your Github repo to it. Since our Model is Huge and to upload huge files on Github we need to use Gitlfs to deploy our model on github as i have done and when connecting your repository to Heroku do not forget to add Gitlfs buildpack otherwise you will receive error rather than a web app. You can cofigure your repository for Continuos Integration and than you can enble continuous integration on heroku server which will update your app as soon as your repository is updated.

## Assumptions-:
Since model is trained for specific crops only so it can diagnose those specific crops only.
The List of Crops For which this model will be helpful is:


<img src="static/Screenshot_3.png">

The crop which can be used for diagnosis can only diagnose specific disease for which the model is trained. 
The List of crop diseases on Which Model is trained on is:


<img src="static/Screenshot_2.png">











