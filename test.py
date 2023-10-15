import torch
import torch.nn as nn
from utils.readData import read_dataset
from utils.ResNet import ResNet34



# 设置设备，如果有CUDA GPU，则使用GPU，否则使用CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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

