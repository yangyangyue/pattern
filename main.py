#
# Written by Lily <3477408934@qq.com>
# 接收用户参数, 调用算法, 训练模型, 计算指标
# github: https://github.com/yangyangyue/pattern.git
# ps: MNIST统一使用torch内置的数据集


import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from classifier import KNNClassifier, CNNClassifier, RandomForestClassifier
import click


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    # 如果没有子命令的话，则按默认配置依次执行RF, KNN和CNN算法
    if ctx.invoked_subcommand is None:
        rf.main(standalone_mode=False)
        knn.main(standalone_mode=False)
        cnn.main(standalone_mode=False)


@cli.command()
@click.option('--k', default=10, help="KNN分类中的K值")
@click.option('--count', default=100, help="KNN分类测试样本数量")
def knn(k, count):
    """k近邻分类"""
    # 加载数据
    train_data = datasets.MNIST(root="./data/", train=True, download=True)
    test_data = datasets.MNIST(root="./data/", train=False, download=True)
    assert 0 < k, "KNN近邻数量应大于0"
    assert 0 < count <= len(test_data), f"KNN测试样本数量应在0到{len(test_data)}之间"
    # 实例化分类器
    knn_classifier = KNNClassifier(k)
    # 对测试样本进行分类，并计算准确率指标
    t = 0

    print(f"KNN测试中, 进度:", end="     ", flush=True)
    for idx in range(count):
        y = knn_classifier(train_data.data.numpy().astype(np.float32), train_data.targets.numpy(),
                           test_data.data[idx].numpy().astype(np.float32))
        t += y == test_data.targets[idx].item()
        print(f"\b\b\b\b{(idx + 1) / count:.2f}", end="", flush=True)
    print(f"\rKNN测试完成, 测试样本数：{count}，测试准确率：{t / count:.4f}")


@cli.command()
@click.option('--his/--no-his', default=True, help="是否加载历史模型")
@click.option('--checkpoint/--no-checkpoint', "check", default=False, help="是否加载检查点")
@click.option('--lr', default=1e-4, help="CNN训练学习率")
@click.option('--drop', default=30, help="CNN训练学习率衰减周期")
@click.option("--epochs", default=100, help="CNN训练轮数")
def cnn(his, check, lr, drop, epochs):
    """卷积神经网络分类"""
    # 如果有gpu则使用gpu， 否则使用cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载MNIST数据
    train_data = datasets.MNIST(root="./data/", train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.MNIST(root="./data/", train=False, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    # 模型、参数及优化器等
    classifier = CNNClassifier().to(device)
    params = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.99))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=drop, gamma=0.1)

    if his:
        # 加载已训练模型
        checkpoint = torch.load('./his/model.pth')
        classifier.load_state_dict(checkpoint['model_state'])
        # 测试 & 计算准确率指标
        print("CNN模型加载成功, 测试中, 进度:", end="     ", flush=True)
        test_acc = 0
        i = 0
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            pred = classifier(imgs)
            test_acc += (torch.argmax(pred, dim=1) == targets).sum()
            i += 1
            print(f"\b\b\b\b{i / len(test_loader):.2f}", end="", flush=True)
        print(f"\rCNN测试完成, 测试样本数: {len(test_data)}, 测试准确率: {test_acc / len(test_data):.4f}")
    else:
        if check:
            # 加载检查点
            checkpoint = torch.load('./his/model.pth')
            classifier.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start = checkpoint['epoch'] + 1
            print("CNN检查点已加载")
        else:
            start = 0

        # 按轮训练，测试
        for epoch in range(start, epochs):
            loss_fn = nn.CrossEntropyLoss()
            train_loss = 0
            train_acc = 0
            test_loss = 0
            test_acc = 0
            print(f"CNN第{epoch}轮训练开始", end='...')
            for imgs, targets in train_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                pred = classifier(imgs)
                loss = loss_fn(pred, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 指标
                train_loss += loss
                train_acc += (torch.argmax(pred, dim=1) == targets).sum()
            # 更新学习率
            lr_scheduler.step()
            print(f"\b\b\b\b\b结束, 训练误差： {train_loss / len(train_loader)}; 训练准确率：{train_acc / len(train_data)}")
            print(f"CNN第{epoch}轮测试开始", end='...')
            with torch.no_grad():
                for imgs, targets in test_loader:
                    pred = classifier(imgs)
                    test_loss += loss_fn(pred, targets)
                    test_acc += (torch.argmax(pred, dim=1) == targets).sum()
            print(f"\b\b\b\b\b结束, 测试误差：{test_loss / len(test_loader)}; 测试准确率：{test_acc / len(test_data)}")

            # 每轮保存一次模型、优化器及当前轮数，任务比较简单，不需要特别地保存训练效果最佳的模型
            checkpoint = {'model_state': classifier.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict(), }
            torch.save(checkpoint, './his/model.pth')


@cli.command()
@click.option('--his/--no-his', default=True, help="是否加载已训练模型")
@click.option('--tree', "n_trees", default=10, help="随机森林中决策树数量")
def rf(his, n_trees):
    """
    随机森林分类
    """
    # 加载数据
    train_data = datasets.MNIST(root="./data/", train=True, download=True)
    test_data = datasets.MNIST(root="./data/", train=False, download=True)

    if his:
        # 加载历史模型
        print("RF模型加载成功, 测试中， 进度:", end="     ")
        with open("./his/rf.pth", "rb") as f:
            classifier = pickle.load(f)
    else:
        # 训练 & 保存模型
        with open("./his/rf.pth", "wb") as f:
            classifier = RandomForestClassifier(n_trees=n_trees)
            classifier.fit(train_data.data.numpy(), train_data.targets.numpy())
            pickle.dump(classifier, f)
    # 测试
    t = 0
    i = 0
    for sample, target in test_data:
        t += target == classifier(np.array(sample))
        i += 1
        print(f"\b\b\b\b{i / len(test_data):.2f}", end="")
    print(f"\rRF测试完成, 测试样本数: {len(test_data)}, 测试准确率: {t / len(test_data):.4f}")


if __name__ == '__main__':
    cli()
