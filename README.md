# LAFNET

## Dataset: Kaggle FlameVison
It's an opensource dataset, we can download it from easily from this website: https://www.kaggle.com/datasets/anamibnjafar0/flamevision

## How to build the LAFNET on ultralytics yolov5 environment
### The model is base on ultralytics YOLOv5 framework of version 7.0 
https://github.com/ultralytics/yolov5

1. Download YOLOv5 environment
   
   `git clone https://github.com/ultralytics/yolov5.git`
   
2. Build LAFNET model
  Move the entire content of LAFNET.py into common.py, and add it to the yolov5-7.0/models/common.py file by simply copying it directly.
   


2. modify yolo.py.

   LCGBlock is abbreviated as LGBLOCK, which means LG Block.
   
 ```
# In file models//yolo.py
# Find line 28 and add "LCGBlock" there.

from models.common import (
C3,
C3SPP,
LCGBlock
)

# Find code blow, and  add "LCGBlock":
#396行
if m in {
    Conv,
    GhostConv,
    LCGBlock
# 421
if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x, LCGBlock}:
 ```
3. Use LAFNET.yaml as configuration  file for training and detection.

4.The training and inference methods are the same as YOLOv5. Please refer to its usage instructions.



## How to deploy LAFNET on Nvidia edge computing device.

LAFNET can be deployed by tensorRT_Pro-YOLOv8 framework https://github.com/Melody-Zhou/tensorRT_Pro-YOLOv8
or its base repo https://github.com/shouxieai/tensorRT_Pro

1. Install environmental dependency tensorRT_Pro.
2. Convert the pytorch pt and deploy follow the repo tutorial.



## Ongoing efforts to build the repo ...

   
