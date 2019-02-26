import numpy as np
from PIL import Image
import os
import tqdm

BASEDIR = os.path.join(os.path.dirname(__file__), './')

# standard output format
SPACE = 35

# tqdm parameter
UNIT_SCALE = True


def load(set_='train_resize', label=True):
    x = []
    y = []
    path = BASEDIR+'/dataset/'+set_
    for name in tqdm.tqdm(
            os.listdir(BASEDIR+'/dataset/'+set_),
            desc='{:{}}'.format('Load dataset', SPACE),
            unit_scale=UNIT_SCALE):
        img = Image.open(path+'/'+name)
        img = np.asarray(img, np.float64)
        class_ = name.split('_')[0]
        if not label:
            class_ = class_[:class_.rfind('.')] + '.png'
        x.append(img)
        y.append(class_)
    x = np.asarray(x)
    x = mean_substraction(x)
    if label:
        y = np.asarray(y, np.int32)
    return x, y


# mean substraction by RGB
def mean_substraction(x):
    size_ = x.shape[0]
    mean = [141.45639998, 136.75046567, 119.34598043]
    std = [71.96843246, 70.93090444, 75.99979494]
    for j in range(size_):
        for i in range(3):
            x[j][:, :, i] = (x[j][:, :, i] - mean[i]) / (std[i] + 1e-7)
    return x


def mean_std(x):
    size_ = x.shape[0]
    mean_ = np.array([0.0, 0.0, 0.0])
    std_ = np.array([0.0, 0.0, 0.0])
    p = 0
    for i in range(size_):
        h, w = x[i].shape[0:2]
        p += h*w

    for i in range(size_):
        mean_[0] += np.sum(x[i][:, :, 0])/p
        mean_[1] += np.sum(x[i][:, :, 1])/p
        mean_[2] += np.sum(x[i][:, :, 2])/p

    for i in range(size_):
        std_[0] += np.sum(((x[i][:, :, 0]-mean_[0])**2)/p)
        std_[1] += np.sum(((x[i][:, :, 1]-mean_[1])**2)/p)
        std_[2] += np.sum(((x[i][:, :, 2]-mean_[2])**2)/p)
    std_ = np.sqrt(std_)

    print('{:{}}: {}'.format('Mean', SPACE, mean_))

    print('{:{}}: {}'.format('Std', SPACE, std_))


def main():
    x_train, y_train = load('train_resize')
    x_test, y_test = load('val_resize')


if __name__ == '__main__':
    main()
