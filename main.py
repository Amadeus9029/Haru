import yaml
import os
from data import *
from model import *
from torchvision import transforms
from torch.utils.data import DataLoader
from Haru import Haru
import json


# 作者：上海-悠悠 交流QQ群：588402570

def main():
    # 获取当前脚本所在文件夹路径
    curPath = os.path.dirname(os.path.realpath(__file__))
    # 获取yaml文件路径
    yamlPath = os.path.join(curPath, "application.yml")

    # open方法打开直接读出来
    f = open(yamlPath, 'r', encoding='utf-8')
    cfg = f.read()
    print(type(cfg))  # 读出来是字符串
    print(cfg)

    d = yaml.safe_load(cfg)  # 用load方法转字典
    print(d)
    print(type(d))


def load_data(mode, aug=False):
    # 多种类型的数据集的类供用户选择   只做这个
    # transforms
    # 路径
    # 模式
    # 自定义的类
    # torch内的类
    train_dir = r'E:\研究生\MyData\filelists'
    val_dir = r'E:\研究生\MyData\filelists'
    if mode == 'train':
        dataset = RainHazeImageDataset(train_dir, 'train',
                                       aug=aug,
                                       transform=transforms.Compose([ToTensor()]))
        shuff = True
    elif mode == 'val':
        dataset = RainHazeImageDataset(val_dir, 'val',
                                       transform=transforms.Compose([ToTensor()]))
        shuff = False
    else:
        dataset = None
        shuff = False
        print('Undefined mode', mode)
        exit()
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=shuff,
                             drop_last=True,
                             num_workers=0)
    return data_loader


def load_train_data(data):
    def wrapper(func):
        def deco(*args, **kwargs):
            load_data('train')
            # 真正执行函数的地方
            func()

        return deco

    return wrapper


@load_train_data("data from demo")
def train_model():
    print("hello world")


if __name__ == '__main__':
    # 获取当前脚本所在文件夹路径
    curPath = os.path.dirname(os.path.realpath(__file__))
    # 获取yaml文件路径
    yamlPath = os.path.join(curPath, "application.yml")

    # open方法打开直接读出来
    f = open(yamlPath, 'r', encoding='utf-8')
    cfg = f.read()

    config = yaml.safe_load(cfg)  # 用load方法转字典
    print(config)
    print(type(config))
    app = Haru(config).test()
