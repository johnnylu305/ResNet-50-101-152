import os
import PIL
import random
from PIL import Image


random.seed(22)
dir__ = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed",
         "Common wheat", "Fat Hen", "Loose Silky-bent", "Maize",
         "Scentless Mayweed", "Shepherds Purse", "Small-flowered Cranesbill",
         "Sugar beet"]

dir_ = "./dataset/train"

path = './dataset/train_resize'
if not os.path.isdir(path):
    os.mkdir(path)

path2 = './dataset/val_resize'
if not os.path.isdir(path2):
    os.mkdir(path2)

path3 = './dataset/val2_resize'
if not os.path.isdir(path3):
    os.mkdir(path3)

# resize training set
for l, j in enumerate(dir__):
    path_ = './'+dir_+'/'+j
    for k in os.listdir(path_):
        open_path = path_+'/'+k
        save_path = path+'/'+str(l)+'_'+k[:k.rfind('.')]+'.jpg'
        save_path2 = path2+'/'+str(l)+'_'+k[:k.rfind('.')]+'.jpg'
        save_path3 = path3+'/'+str(l)+'_'+k[:k.rfind('.')]+'.jpg'

        img = Image.open(open_path)
        w, h = img.size
        # training set
        if random.random() <= 0.8:
            if abs(w-256) > abs(h-256):
                img = img.resize((int(w*(256.0/h)), 256), PIL.Image.LANCZOS)
            else:
                img = img.resize((256, int(h*(256.0/w))), PIL.Image.LANCZOS)
            img.convert('RGB').save(save_path)
        # validation set
        else:
            img1 = img.resize((224, 224), PIL.Image.LANCZOS)
            img1.convert('RGB').save(save_path2)
            if abs(w-256) > abs(h-256):
                img2 = img.resize((int(w*(256.0/h)), 256), PIL.Image.LANCZOS)
            else:
                img2 = img.resize((256, int(h*(256.0/w))), PIL.Image.LANCZOS)
            img2.convert('RGB').save(save_path3)

dir_ = "./dataset/test"

path = './dataset/test_resize'
if not os.path.isdir(path):
    os.mkdir(path)

# resize testing set

path_ = './'+dir_+'/'
for k in os.listdir(path_):
    open_path = path_+'/'+k
    save_path = path+'/'+k[:k.rfind('.')]+'.jpg'

    img = Image.open(open_path)
    w, h = img.size
    img = img.resize((224, 224), PIL.Image.LANCZOS)
    img.convert('RGB').save(save_path)
