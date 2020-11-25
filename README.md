# CS_T0828_HW2

Code for Selected Topics in Visual Recognition
using Deep Learning(2020 Autumn) HW2.

This code is based on [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet), modified for [Street View House Numbers](https://drive.google.com/drive/u/1/folders/1Ob5oT9Lcmz7g5mVOcYH3QugA7tV3WsSl) training Data

# Hardware
The following specs were used to create the original solution.
- ubuntu 16.04 LTS
- Intel® Core™ i9-10900 CPU @ 3.70GHz x 20
- 2x RTX 2080 Ti

## Reproducing Submission
To Reproduct the submission, do the folowing steps
1. [Framework Download and Setting](#Framework-Download-and-Setting)
2. [Dataset Preparation](#Dataset-Preparation)
3. [Training](#Training)
4. [Testing](#Testing)
5. [Output postprocessing](#Output-postprocessing)

## Requirements
My Environment settings are below:
* Python = 3.7.9
* pandas = 1.1.3
* opencv = 4.4.0

In this work, I used yolov4 framework from [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet), so you should also followed the instruction of AlexeyAB's darknet:
```
Windows or Linux
CMake >= 3.12
CUDA >= 10.0
OpenCV >= 2.4
cuDNN >= 7.0
GPU with CC >= 3.0
```
## Framework Download and Setting
Download AlexeyAB's darknet from https://github.com/AlexeyAB/darknet
and unzip the file.
Or use commandline `$git clone https://github.com/AlexeyAB/darknet`

Follow the [instruction](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux-using-make) to modify the `Makefile`. For my work, it should like this:
* `GPU=1` to train with GPU
* `CUDCNN=1` build with cuDNN to accelerate training
* `OPENCV=1`  to build with OpenCV
* `OPENMP=1`
* `LIBSO=1`
Use `~/darknet$ make` command.
## Dataset Preparation
### Training and Testing Data
* Download directory `my` and put it under darknet.
* Download the pre-trained weights-file (162 MB): [yolov4.conv.137](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137) and put it in `my`
* Put dataprecess.py under darknet.
* Put training and testing data in data directory

now the files should structured like this:
```
darknet
    +- backup * models generated will be saved here*
    +- data
    |    +- test
    |    | 1.png
    |    | ...
    |    +-train
    |    | 1.png
    |    | 2.png
    |    |...
    +- my
    |    | obj.data
    |    | obj.names 
    |    | test.txt
    |    | train.txt
    |    | yolov4.conv.137
    |    | yolov4-obj.cfg
    |    
    darknet
    dataprocess.py
    ...
```
Use command`$ python3 datapreprocess.py preprocess` to start making .txt files for darknet training.
After preprocessing, there will have `<img_name>.txt` in `train` director for each training image, `train.txt` and `test.txt` in `my` director which contains path to every training/testing image which likes:
```
data/train/1.png
data/train/2.png
...
```
### Config
Modify `obj.data` and `obj.names`
* obj.data contains paths to relative files
```
classes = 10 (for digit 0~9)
train = my/train.txt (path from darknet to train.txt)
valid = my/test.txt  (path from darknet to test.txt)
names = my/obj.names (path from darknet to obj.names)
backup = backup/ (backup directory, models generated during testing will be saves here)
```
* obj.names contains <num_classes> lines, each line for one class
```
0
1
2
3
4
5
6
7
8
9
```
Folow [Instruction] (https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)modify `yolov4-obj.cfg` into right form:
- batch=64 (could be smaller if CUDA out of memore occurs)
- subdivisions=64
- width=416
- height=416
- learning_rate=0.001
- max_batches=33402 (it should > class*2000, not less than number of training images and not less than 6000)
- steps=26000,30000 (80% and 90% of max_batches)

## Training
To train the model, run folowed command
`$ ./darknet detector train my/obj.data my/yolov4-obj.cfg my/yolov4.conv.137 -dont_show -gpus 0,1`
It will generate `yolov4-obj_<Epoch>.weights` in the backup directory.
If `CUDA Error: out of memory` occurs, try smaller batch.

During training, you can check chart.png for training loss.
![](https://i.imgur.com/HbkDi6v.png)

## Testing
After training, you can find `yolov4-obj_final.weights` in the backup director, use command to test on testing images
```
./darknet detector test my/obj.data my/yolov4-obj.cfg backup/yolov4-obj_final.weights -ext_output -dont_show -out result.json < my/test.txt -gpus 2 -thresh <threshold>
```

It will generate `result.json` in directory darknet
## Output postprocessing
To generate submisson.json file:
![](https://i.imgur.com/Id5qci4.png)

Run the command`$ python3 datapreprocess.py postprocess`
It will read `result.json` and `img_wh.txt`, then generate json file in the required format.
![](https://i.imgur.com/uIhoLVJ.png)
