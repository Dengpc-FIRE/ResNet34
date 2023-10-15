import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils.readData import read_dataset
from utils.ResNet import ResNet34


# 设置设备，如果有CUDA GPU，则使用GPU，否则使用CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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
