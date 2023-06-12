# Global Convolutional Neural Networks With Self-Attention for Fisheye Image Rectification

## Introduction
This is a TensorFlow implementation for Global Convolutional Neural Networks With Self-Attention for Fisheye Image Rectification.  
This work has been published in <a href="https://ieeexplore.ieee.org/document/9980359">IEEE Access</a>.

![image](https://github.com/byunghyun23/GSAFE/blob/main/assets/fig1.png)

## Architecture
![image](https://github.com/byunghyun23/GSAFE/blob/main/assets/fig2.png)

## Installation
1. Clone the project
```
git clone https://github.com/byunghyun23/GSAFE
```
2. Install the prerequisites
```
pip install -r requirements.txt
```

## Dataset
For training the model, you need to download the dataset [here](https://drive.google.com/file/d/1lRsQBmwZyri6-reNWHbR9AzS3cKiiu78/view?usp=share_link) or full [Places2](http://places2.csail.mit.edu/download.html).  
Then, move the downloaded images to
```
--data/images
```
Run
```
python data_generator.py
python data_splitter.py
```
to distort and split the fisheye dataset. 
The distorted fisheye images will be placed in 
```
--data/distorted
```
and split fisheye images will be placed in
```
--data/train_input
--data/train_target
--data/test_input
--data/test_target
```

## Train
```
python train.py
```

## Test
```
python test.py
```

## Rectification
You can use your fisheye image.
Before Start, make sure that the fisheye image have been placed in
```
--sample
```
Run
```
python calib.py
```
After rectification, the results will be placed in
```
--results
```
