import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from .CutOut import cut

# 设置设备
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# 用于数据加载的子进程数量
num_workers = 0
# 每批加载图像的数量
batch_size = 16
# 用作验证集的训练集百分比
valid_size = 0.2


def read_dataset(batch_size=16, valid_size=0.2, num_workers=0, pic_path='dataset'):
    """
    batch_size: 每批加载图像的数量
    valid_size: 用作验证集的训练集百分比
    num_workers: 用于数据加载的子进程数量
    pic_path: 图像数据的路径
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 在四周填充0的基础上，随机裁剪图像为32*32
        transforms.RandomHorizontalFlip(),  # 以一半的概率水平翻转图像
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 使用的R、G、B通道的均值和标准差用于归一化
        cut(n_holes=1, length=16),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    train_data = datasets.CIFAR10(pic_path, train=True,
                                  download=True, transform=transform_train)
    valid_data = datasets.CIFAR10(pic_path, train=True,
                                  download=True, transform=transform_test)
    test_data = datasets.CIFAR10(pic_path, train=False,
                                 download=True, transform=transform_test)

    # 获取用于验证的训练数据的索引。创建一个随机打乱的索引列表
    num_train = len(train_data)
    indices = list(range(num_train))
    # 随机打乱索引
    np.random.shuffle(indices)
    # 划分比例
    split = int(np.floor(valid_size * num_train))
    # 分割数据为训练数据和验证数据
    train_idx, valid_idx = indices[split:], indices[:split]

    # 为获取训练和验证批次定义采样器
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # 准备数据加载器（结合数据集和采样器）
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              num_workers=num_workers)

    return train_loader, valid_loader, test_loader