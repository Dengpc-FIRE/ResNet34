import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class cut(object):

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            # (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,**kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,  # 残差结构的类别ResNet18采用BasicBlock
                 blocks_num: list,  # 列表,每层残差结构的个数
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x



def ResNet18(num_class=1000,include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2],num_classes=num_class,include_top=include_top)

def ResNet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)




import time
t1=time.time()





# 设置设备，如果有CUDA GPU，则使用GPU，否则使用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# 读取数据
batch_size = 128
train_loader, valid_loader, test_loader = read_dataset(batch_size=batch_size, pic_path='dataset')

# 加载模型（使用预训练模型，修改最后一层，固定之前的权重）
n_class = 10
model = ResNet34()
"""
ResNet18网络的7x7降采样卷积和池化操作容易丢失一部分信息,
所以在实验中我们将7x7的降采样层和最大池化层去掉,替换为一个3x3的降采样卷积,
同时减小该卷积层的步长和填充大小
"""
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, n_class)  # 将最后的全连接层修改
model = model.to(device)

# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss().to(device)

# 开始训练
n_epochs = 1
valid_loss_min = np.Inf  # 用于跟踪验证集损失的变化
accuracy = []
lr = 0.1
counter = 0
for epoch in tqdm(range(1, n_epochs + 1)):

    # 用于跟踪训练和验证损失
    train_loss = 0.0
    valid_loss = 0.0
    total_sample = 0
    right_sample = 0

    # 动态调整学习率
    if counter / 10 == 1:
        counter = 0
        lr = lr * 0.5

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    ###################
    # 训练集上的模型 #
    ###################
    model.train()  # 启用模型的训练模式，启用批量归一化和丢弃
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        # 清空所有优化变量的梯度
        optimizer.zero_grad()
        # 正向传播：通过将输入传递给模型来计算预测输出
        output = model(data).to(device)
        # 计算批量损失
        loss = criterion(output, target)
        # 反向传播：计算相对于模型参数的损失梯度
        loss.backward()
        # 执行单次优化步骤（参数更新）
        optimizer.step()
        # 更新训练损失
        train_loss += loss.item() * data.size(0)

    ######################
    # 验证集上的模型 #
    ######################
    model.eval()  # 验证模型
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)
        # 正向传播：通过将输入传递给模型来计算预测输出
        output = model(data).to(device)
        # 计算批量损失
        loss = criterion(output, target)
        # 更新平均验证损失
        valid_loss += loss.item() * data.size(0)
        # 将输出概率转换为预测类别
        _, pred = torch.max(output, 1)
        # 比较预测与真实标签如果它们相等，则 correct_tensor 中相应位置的元素将设置为 True
        correct_tensor = pred.eq(target.data.view_as(pred))
        total_sample += batch_size
        for i in correct_tensor:
            if i:
                right_sample += 1
    print("准确率:", 100 * right_sample / total_sample, "%")
    accuracy.append(right_sample / total_sample)

    # 计算平均损失
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)

    # 显示训练集和验证集的损失
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

    # 如果验证集损失减小，保存模型
    if valid_loss <= valid_loss_min:
        print('验证损失减小 ({:.6f} --> {:.6f}). 正在保存模型...'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), '.resnet18_cifar10.pt')
        valid_loss_min = valid_loss
        counter = 0
    else:
        counter += 1



t2=time.time()
print("花费时间为:{}".format(t2-t1))
print("下面进行测试:")




# 设置设备，如果有CUDA GPU，则使用GPU，否则使用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 类别数量，通常对应数据集中的类别数量
n_class = 10

# 测试数据集的批量大小
batch_size = 100

# 从read_dataset函数中加载训练、验证和测试数据加载器
train_loader, valid_loader, test_loader = read_dataset(batch_size=batch_size, pic_path='dataset')

# 创建一个名为model的ResNet18模型的实例
model = ResNet34()

# 修改模型的第一个卷积层
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

# 修改模型的全连接层，以适应你的数据集中的类别数量
model.fc = torch.nn.Linear(512, n_class)

# 加载预训练的模型权重
model.load_state_dict(torch.load('.resnet18_cifar10.pt'))

# 将模型移动到GPU（如果可用）或CPU上进行推断
model = model.to(device)

# 初始化总样本数
total_sample = 0

# 初始化正确分类的样本数
right_sample = 0

# 将模型切换到评估模式
model.eval()

# 通过test_loader迭代测试数据集
for data, target in test_loader:
    # 将数据和目标移动到指定的设备
    data = data.to(device)
    target = target.to(device)

    # 通过模型前向传播计算预测输出
    output = model(data).to(device)

    # 将输出概率转换为预测类别
    _, pred = torch.max(output, 1)

    # 比较预测值与真实标签，以确定是否分类正确
    correct_tensor = pred.eq(target.data.view_as(pred))

    # 更新总样本数和正确分类的样本数
    total_sample += batch_size
    for i in correct_tensor:
        if i:
            right_sample += 1

# 计算并打印准确率
print("准确率:", 100 * right_sample / total_sample, "%")

