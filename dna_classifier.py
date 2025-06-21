import numpy as np
from collections import Counter
from math import log2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from graphviz import Digraph
from PIL import Image as PilImage

# 定义树节点类
class Node:
    def __init__(self, attribute=None, threshold=None, left=None, right=None, *, value=None):
        self.attribute = attribute  # 当前节点用于分裂的属性
        self.threshold = threshold  # 用于分裂的阈值，这里不使用
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.value = value  # 叶节点的值

# 定义ID3决策树类
class ID3:
    def __init__(self):
        self.root = None  # 初始化根节点

    # 训练模型
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    # 预测新数据
    def predict(self, X):
        return [self._traverse_tree(x, self.root) for x in X]

    # 递归地生长决策树
    def _grow_tree(self, X, y):
        num_samples, num_features = X.shape

        # 如果所有样本属于同一类别，则返回叶节点
        if len(set(y)) == 1:
            return Node(value=y[0])

        # 如果没有更多的特征可用于分裂，返回叶节点
        if num_features == 0:
            return Node(value=Counter(y).most_common(1)[0][0])

        # 选择最优的分裂属性
        best_attr = self._best_attribute(X, y)
        if best_attr is None:
            return Node(value=Counter(y).most_common(1)[0][0])

        # 创建当前节点并继续生长左右子树
        node = Node(attribute=best_attr)
        left_indices = X[:, best_attr] == 1
        right_indices = X[:, best_attr] == 0

        node.left = self._grow_tree(X[left_indices], y[left_indices])
        node.right = self._grow_tree(X[right_indices], y[right_indices])
        return node

    # 选择最佳的分裂属性
    def _best_attribute(self, X, y):
        num_samples, num_features = X.shape
        if num_samples <= 1:
            return None

        base_entropy = self._entropy(y)  # 计算当前集合的信息熵
        best_info_gain = -1  # 初始化最佳信息增益
        best_attr = None  # 初始化最佳属性

        for attr in range(num_features):
            left_indices = X[:, attr] == 1
            right_indices = X[:, attr] == 0

            if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                continue

            # 计算分裂后的信息熵
            left_entropy = self._entropy(y[left_indices])
            right_entropy = self._entropy(y[right_indices])
            info_gain = base_entropy - (
                    np.sum(left_indices) / num_samples * left_entropy +
                    np.sum(right_indices) / num_samples * right_entropy
            )

            # 更新最佳信息增益和最佳属性
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_attr = attr

        return best_attr

    # 计算信息熵
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * log2(p) for p in ps if p > 0])

    # 遍历决策树进行预测
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.attribute] == 1:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    # 决策树可视化
    def visualize_tree(self, node, feature_names):
        if node is None:
            return None

        # 创建一个有向图对象，设置布局为从上到下
        dot = Digraph(format='png')
        dot.graph_attr['rankdir'] = 'TB'  # TB: Top to Bottom, LR: Left to Right
        dot.node_attr['shape'] = 'box'  # 设置节点形状为方框

        def add_nodes_edges(node, parent=None):
            if node.value is not None:
                label = f"Class: {node.value}"
            else:
                label = f"{feature_names[node.attribute]}"
            dot.node(str(id(node)), label=label)
            if parent:
                dot.edge(str(id(parent)), str(id(node)))
            if node.left:
                add_nodes_edges(node.left, node)
            if node.right:
                add_nodes_edges(node.right, node)

        add_nodes_edges(node)
        return dot

def read_data(file_path):
    try:
        # 尝试使用空格分隔符读取数据
        data = pd.read_csv(file_path, header=None, sep=' ')
    except:
        try:
            # 尝试使用逗号分隔符读取数据
            data = pd.read_csv(file_path, header=None, sep=',')
        except:
            # 尝试使用制表符分隔符读取数据
            data = pd.read_csv(file_path, header=None, sep='\t')
    return data

if __name__ == '__main__':
    train_path = 'dna.data'
    train_data = read_data(train_path)

    # 最后一列是标签，其余是特征
    train_data.iloc[:, -1] = train_data.iloc[:, -1].astype(str)
    print(train_data.head())
    labels_train = train_data.iloc[:, -1].str.replace(';', '').astype(int)
    print(labels_train)

    X = train_data.iloc[:, :-1].values
    y = labels_train.values
    print("y_train:", y)
    # 打印训练集的基本信息
    print("训练集:", Counter(y))

    # 测试集
    test_path = 'dna.test'
    test_data = read_data(test_path)

    # 处理测试数据标签列，去除分号并转换为整数
    test_data.iloc[:, -1] = test_data.iloc[:, -1].astype(str)
    labels_test = test_data.iloc[:, -1].str.replace(';', '').astype(int)

    # 最后一列是标签，其余是特征
    X_test = test_data.iloc[:, :-1].values
    y_test = labels_test.values
    print("y_test:", y_test)
    # 打印测试集的基本信息
    print("测试集：", Counter(y_test))

    # 创建ID3模型并训练
    model = ID3()
    model.fit(X, y)

    # 使用测试集进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Predictions:", y_pred)
    print("Accuracy:", accuracy)

    # 模型性能可视化：混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # 生成特征名称（假设特征为0到倒数第二列的索引）
    feature_names = [str(i) for i in train_data.columns[:-1]]

    # 可视化决策树
    dot = model.visualize_tree(model.root, feature_names)
    dot.render("decision_tree", format="png")

    # 显示决策树图像
    img = PilImage.open("decision_tree.png")
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
