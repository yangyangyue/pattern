#
# Written by Lily <chengyangli@ncepu.edu.cn>
# 实现RF分类器, KNN分类器和CNN分类器
# github: https://github.com/yangyangyue/pattern.git


import random
from collections import Counter

import numpy as np
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self):
        """
        初始化CNN分类器
        考虑到mnist数据集并不复杂，所以使用的CNN分类器也很简单，只包括一个简单的主干特征提取网络以及一个将输出映射为目标形状(10,)的全连接网络
        """
        super(CNNClassifier, self).__init__()
        self.backbone = nn.Sequential(
            self.__make_layer(1, 32),
            self.__make_layer(32, 64)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )

    @staticmethod
    def __make_layer(in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (3, 3)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, (3, 3)),
            nn.BatchNorm2d(out_channel),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class KNNClassifier:
    def __init__(self, k: int):
        self.k = k
        self.n_class = 10

    def __call__(self, train_data, train_targets, img):
        # 计算输入与训练集中各样本的距离
        dis = np.sum((train_data - img) ** 2, axis=(1, 2)) ** 0.5
        # 统计前k个近邻中各个类别的数量
        cls = np.zeros((self.n_class,))
        for idx in np.argpartition(dis, self.k)[:self.k]:
            cls[train_targets[idx]] += 1
        return np.argmax(cls)


class RandomForestClassifier:
    # 大部分参数的设置其实没必要暴漏给用户，自己指定就行
    def __init__(self, n_trees=10, max_depth=-1, min_samples_split=2, min_samples_leaf=1, sample_rate=0.3):
        self.n_class = 10
        self.n_trees = n_trees
        self.trees = []
        self.max_depth = max_depth if max_depth != -1 else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.sample_rate = sample_rate

    def fit(self, samples, targets):
        """
        训练随机森林
        :param samples: 训练集样本, shape:(m, 28, 28)
        :param targets: 训练集标签, shape:(m,)
        """
        samples = samples.reshape((len(samples), -1))
        print("随机森林训练中, 进度:", end=" 0.00", flush=True)
        for i in range(self.n_trees):
            # 随机生成子数据集的索引，用于构造训练每颗树的数据集
            idxs = random.sample(range(len(targets)), int(len(targets) * self.sample_rate))
            self.trees.append(self.__build_tree(samples[idxs], targets[idxs], 0))
            print(f"\b\b\b\b{(i+1) / self.n_trees:.2f}", end="", flush=True)
        print(f"\r随机森林训练完成, 决策树数：{self.n_trees}")

    def __build_tree(self, samples, targets, depth):
        """
        构造一颗决策树，决策树是基于Gini指数的二叉树，使用递归的方法构造
        :param samples: 样本，shape: (m, 28*28)
        :param targets: 标签，shape: (m,)
        :param depth: 当前深度， 用于限制决策树的深度
        :return: 决策树 
        """
        # 样本全部是同一类别，则停止分裂
        if all(target == targets[0] for target in targets):
            return self.Tree(cls=targets[0])
        # 样本数量小于分裂最低样本数或者深度达到限制，则停止分裂
        if len(targets) < self.min_samples_split or depth >= self.max_depth:
            return self.Tree(cls=Counter(targets).most_common(1)[0][0])
        # 随机生成候选分裂特征集合
        sub_features = random.sample(range(len(samples[0])), int(np.sqrt(len(samples[0]))))
        # 选择分裂特征，虽然样本数很多，但是特征值的话大部分也都是重复的，因此无需基于百分位值计算分裂值
        best_feature = 0
        best_value = 0
        best_gain = 1
        for feature in sub_features:
            # 排序, 去重
            values = np.unique(samples[:, feature])
            # 选择最佳分裂特征和分裂值
            for value in values:
                left_targets = targets[samples[:, feature] <= value]
                right_targets = targets[samples[:, feature] > value]
                if all(samples[:, feature] <= value) or all(samples[:, feature] > value):
                    continue
                gain = self.__gain(left_targets, right_targets)
                if gain < best_gain:
                    best_feature, best_value, best_gain = feature, value, gain

        # 划分后所有样本都被划分到同一子集，则停止分裂
        if all(samples[:, best_feature] <= best_value) or all(samples[:, best_feature] > best_value):
            return self.Tree(cls=Counter(targets).most_common(1)[0][0])

        tree = self.Tree(feature=best_feature, value=best_value)
        tree.left = self.__build_tree(samples[samples[:, best_feature] <= best_value],
                                      targets[samples[:, best_feature] <= best_value], depth + 1)
        tree.right = self.__build_tree(samples[samples[:, best_feature] > best_value],
                                       targets[samples[:, best_feature] > best_value], depth + 1)
        return tree

    def __call__(self, x):
        """
        对一个样本进行分类
        """
        x = x.reshape([-1])
        cls = np.zeros((self.n_class,))
        for c in (tree(x) for tree in self.trees):
            cls[c] += 1
        return np.argmax(cls)

    @staticmethod
    def __gini(targets):
        num = len(targets)
        # 数据集中每个类别的数量，用于计算随机样本属于各个类的概率
        # cls_counts = np.zeros((self.n_class,))
        # 本来是自己统计的， 但是太慢了， 使用counter花费时间大约是之前的1/3
        gini = 1
        for _, count in Counter(targets).most_common():
            gini -= (count / num) ** 2
        # for label in targets:
        #     cls_counts[label] += 1
        # for cls_count in cls_counts:
        #     gini -= (cls_count / num) ** 2
        return gini

    def __gain(self, targets1, targets2):
        return (len(targets1) * self.__gini(targets1) + len(targets2) * self.__gini(targets2)) / (
                len(targets1) + len(targets2))

    class Tree(object):

        def __init__(self, feature=None, value=None, cls=None, left=None, right=None):
            self.feature = feature  # 当前节点的分裂特征
            self.value = value  # 当前节点的分裂特征值
            self.cls = cls  # 当前节点的类别，如果当前节点不是叶子节点，则为None
            self.left = left  # 左子树
            self.right = right  # 右子树

        def __call__(self, x):
            # 叶子节点
            if self.cls is not None:
                return self.cls
            # 小于分裂特征值则走左子树，大于分裂特征值则走右子树
            return self.left(x) if x[self.feature] <= self.value else self.right(x)
