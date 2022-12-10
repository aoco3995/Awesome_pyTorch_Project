

# **Awesome PyTorch Project**
Awesome_pyTorch_Project is a Python library for dealing with image classification and object recognition.
## Github Link
https://github.com/aoco3995/Awesome_pyTorch_Project
## Developers
Dylan Cook, Adam O'Connor, Joey Stolfa
## Design Requirements
Image Classifier shall identify [pikachu, cats, dogs, people, drones]

Image Classifer shall have an overall accuaracy of +75% between the different classes

Object detection shall output a video file placing a bounding box around the given classes and labeling accordingly
## Installation


-Download the Awesome_pyTorch_Project.zip folder

-extract contents

-using command line run classifier,py to further train/view the give CNN

-using command line run Draw_bound_Vid.py to view 

-usinf command line run Video_maker.py to create video from a directory

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required imports.

```bash
pip3 install moviepy
pip3 install numpy
pip3 install opencv-python


(for CUDA)
pip3 install pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
or 
(for CPU)
pip3 install torch torchvision torchaudio


```

## Usage

For image classifier
```python
# Run classifier.py when run loads, trains, and saves a CNN 

# Input request to load csv file
Load[y/n]: 

# Input request to Train csv file weights
Train[y/n]: 

# Input request to save csv file weights
Save[y/n]: 

```
For Object recognition and bounding box 
```python
# modify test video directory in line 62 if using a custom video to desired video location
video = cv2.VideoCapture('Test video\Test_Vid.mp4') 

```

## Acknowledgements

 - [pytorch totorial CNN](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)



## Documentation

Can be found in the Documentation folder containing pydoc html files

