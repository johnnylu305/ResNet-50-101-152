import argparse


def train_parser():
    # create parser
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # number of layers for resnet
    # dafault: 101
    parser.add_argument('--layers', type=str, default='101',
                        help='select number of layers from [50, 101, 152]')
    # number of classes
    # dafault: 12
    parser.add_argument('--classes', type=int, default=12,
                        help='number of classes')
    # batch size
    # dafault: 16
    parser.add_argument('--batch', type=int, default=16,
                        help='batch size')
    # max epoch
    # dafault: 300
    parser.add_argument('--epoch', type=int, default=300,
                        help='max epoch')
    # learning rate
    # dafault: 0.001
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    # load pretrain weights or not
    # dafault: true
    parser.add_argument('--pretrain', type=int, default=1,
                        help='load pretrain weights or not. \
                             [0, 1]=[False, True]')
    # save step for saver
    # dafault: 1
    parser.add_argument('--save', type=int, default=1,
                        help='save step for saver')
    # recover weights and continue training or not
    # dafault: false
    parser.add_argument('--recover', type=int, default=-1,
                        help='recover weights and continue training ot not. \
                             [-1, 0+]=[False, True]')
    # add validation set into training set or not
    # dafault: False
    parser.add_argument('--val', type=int, default=0,
                        help='add validation set into training set or not. \
                             [0, 1]=[False, True]')

    # parsing
    args = parser.parse_args()
    # check valid or not
    if args.layers not in ['50', '101', '152']:
        parser.error('Layers must be selected from [50, 101, 152]')
    if args.recover != -1 and args.pretrain == 1:
        parser.error('Recover and pretrain cannot be true at the same time.')
    return args


def test_parser():
    # create parser
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # number of layers for resnet
    # dafault: 101
    parser.add_argument('--layers', type=str, default='101',
                        help='select number of layers from [50, 101, 152]')
    # number of classes
    # dafault: 12
    parser.add_argument('--classes', type=int, default=12,
                        help='number of classes')
    # batch size
    # dafault: 16
    parser.add_argument('--batch', type=int, default=16,
                        help='batch size')
    # recover weights
    # dafault: 0
    parser.add_argument('--recover', type=int, default=0,
                        help='recover weights')
    # parsing
    args = parser.parse_args()
    # check valid or not
    if args.layers not in ['50', '101', '152']:
        parser.error('Layers must be selected from [50, 101, 152]')
    return args
