# ResNet-50/101/152
There are two types of ResNet in **Deep Residual Learning for Image Recognition**, by Kaiming He et al. One for ImageNet and 
another for CIFAR-10. 

I had implemented the **ResNet-50/101/152 (ImageNet one)** by Python with Tensorflow in this repo. You can train my 
ResNet-50/101/152 without pretrain weights or load the pretrain weights of ImageNet. I had trained and tested my
ResNet-50/101/152 on Kaggle Plant Seedings Classification.


## My Environment
### Environment 1
- Operating System:
  - Arch Linux 4.20.7-1
- Memory
  - 64GB
- CUDA:
  - CUDA V10.0.130 
- CUDNN:
  - CUDNN 7.0.5-2
- GPU:
  - GTX 1070 8G
- Nvidia driver:
  - 390.25
- Python:
  - python 3.6.4
- Python package:
  - tqdm, bs4, opencv-python, pydensecrf, cython...
- Tensorflow:
  - tensorflow-gpu 1.5.0


## Downloading the plant seeding dataset from Kaggle
- [Plant seeding dataset](https://www.kaggle.com/c/plant-seedlings-classification/data)

## Setup Dataset
### My directory structure
```
./ResNet/
├── dataset
├── models
└── pretrain_models
```
### Plant seeding dataset directory structure
```
./train/
├── Black-grass
├── Charlock
├── Cleavers
├── Common Chickweed
├── Common wheat
├── Fat Hen
├── Loose Silky-bent
├── Maize
├── Scentless Mayweed
├── Shepherds Purse
├── Small-flowered Cranesbill
└── Sugar beet
```
- Put train.zip and test.zip (Plant seeding dataset) into {PATH}/ResNet-PreAct/ResNet/dataset/
```
mv {PATH}/train.zip {PATH}/ResNet-PreAct/ResNet/dataset/
mv {PATH}/test.zip {PATH}/ResNet-PreAct/ResNet/dataset/
```
- Unzip train.zip and test.zip to the {PATH}/ResNet-PreAct/ResNet/dataset
```
unzip train.zip
unzip test.zip
```
- Resize and split the dataset into train, val and test
```
python ./dataset/resize.py
```

## Demo (See Usage for more details)
### Download pretrain model training on plant seeding dataset
- [Pretrain model](https://drive.google.com/drive/folders/1BLMyijyADHRtMUzuvk4MPfBeDLAMbuue?usp=sharing)
  - Move files from resnet_v2_50_val to '{PATH}/models/resnet_v2_50'
- Run test
  ```
  python test.py --layers 50 --batch 1 --recover 272  
  ```
- Run train (See Training for more details)
  ```
  python train.py --layers 50 --batch 16 --val 0 --recover 272 --pretrain 0  
  ```
- Performance


| Key | Value | 
| :-------- |:-------- | 
| Method | ResNet50-Preact | 
|Language|Python with Tensorflow|
|Pretrain weight|ImageNet|
|Dataset|20% to validation|
|Best epoch|272|
|Validation set accuracy|96.72%|
|Submission time|2019/1/18|
|Result|96.72%|

- [Other performance](https://hackmd.io/mVVx1qtNSgWwuu_1w9piXg?both)


## Training (See Usage for more details)
### Download pretrain weights of ImageNet
- [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/slim)
  - Put resnet_v2_*.ckpt in 'pretrain_models'
### Train network
- Set parameters
  - set number of layers for ResNet
  - set loading pretrain weights of ImageNet or not
  - set adding validation set into training set or not
  - Check Usage for other parameters 
```
python train.py --layers [50|101|152] --pretrain [0|1] --val [0|1]    
```
- For example
```
python train.py --layers 50 --pretrain 1 --val 0    
```


## Check performance (See Usage for more details)
### Test validation set
```
python val.py --layer [50|101|152] --recover [0+]    
```
- For example
```
python val.py --layer 50 --recover 299    
```


## Testing (See Usage for more details)
### Test network
- Result will be stored as '.csv' file in {PATH}/ResNet/dataset/
```
python test.py --layers [50|101|152] --batch 1 --recover [0+]
```
- For example
```
python test.py --layers 50 --batch 1 --recover 299
```


## Training on your own dataset
- Put the images of training set into {PATH}/ResNet/train/{class_name}/
- If you have testing set, put it into {PATH}/ResNet/test/
- Modify the class name in resize.py


## Usage
### dataset/resize.py
- Resize and split the dataset into train, val, test
### eval_.py
- Compute the accuracy of the result
- May be called by train.py and val.py
### load_data.py
- Load and preprocess dataset
- May be called by train.py, val.py and test.py
### parser.py
- Parse the command line argument
- May be called by train.py, val.py and test.py
### resnet.py
- The implementation of the network
### test.py
- Test the testing set and store the result in the 'dataset' as .csv file
```
usage: test.py [-h] [--layers LAYERS] [--classes CLASSES] [--batch BATCH]
               [--recover RECOVER]

optional arguments:
  -h, --help         show this help message and exit
  --layers LAYERS    select number of layers from [50, 101, 152] (default:
                     101)
  --classes CLASSES  number of classes (default: 12)
  --batch BATCH      batch size (default: 16)
  --recover RECOVER  recover weights (default: 0)
```
### train.py
- Train the network
```
usage: train.py [-h] [--layers LAYERS] [--classes CLASSES] [--batch BATCH]
                [--epoch EPOCH] [--lr LR] [--pretrain PRETRAIN] [--save SAVE]
                [--recover RECOVER] [--val VAL]

optional arguments:
  -h, --help           show this help message and exit
  --layers LAYERS      select number of layers from [50, 101, 152] (default:
                       101)
  --classes CLASSES    number of classes (default: 12)
  --batch BATCH        batch size (default: 16)
  --epoch EPOCH        max epoch (default: 300)
  --lr LR              learning rate (default: 0.001)
  --pretrain PRETRAIN  load pretrain weights or not. [0, 1]=[False, True]
                       (default: 1)
  --save SAVE          save step for saver (default: 1)
  --recover RECOVER    recover weights and continue training ot not. [-1,
                       0+]=[False, True] (default: -1)
  --val VAL            add validation set into training set or not. [0,
                       1]=[False, True] (default: 0)
```
### val.py
- Test the validation set
```
usage: val.py [-h] [--layers LAYERS] [--classes CLASSES] [--batch BATCH]
              [--recover RECOVER]

optional arguments:
  -h, --help         show this help message and exit
  --layers LAYERS    select number of layers from [50, 101, 152] (default:
                     101)
  --classes CLASSES  number of classes (default: 12)
  --batch BATCH      batch size (default: 16)
  --recover RECOVER  recover weights (default: 0)
```


## Reference
- [[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition.](https://arxiv.org/abs/1512.03385)
- [[2] Tensorflow Slim](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py)











