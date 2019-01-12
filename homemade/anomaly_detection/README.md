# 应用高斯分布的异常检测 Anomaly Detection Using Gaussian Distribution

## Jupyter Demos

▶️ [Demo | 异常检测 Anomaly Detection](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/anomaly_detection/anomaly_detection_gaussian_demo.ipynb) - 发现服务器操作参数中的异常，例如`latency`和`threshold`

## 高斯(正态)分布 Gaussian (Normal) Distribution

**正态**(或者**高斯**)**分布**是一个常见的通用连续概率分布。 正态分布在统计学中很重要，并且通常在自然科学和社会科学中用于表示其分布未知的实值随机变量。具有高斯分布的随机变量被称为正态分布并且被称为正态偏差。

如下介绍:

![x-in-R](../../images/anomaly_detection/x-in-R.svg)

如果 _x_ 是 正态分布，它的分布如下图所示：

![Gaussian Distribution](https://upload.wikimedia.org/wikipedia/commons/7/74/Normal_Distribution_PDF.svg)

![mu](../../images/anomaly_detection/mu.svg) - 代表均值,

![sigma-2](../../images/anomaly_detection/sigma-2.svg) - 方差.

![x-normal](../../images/anomaly_detection/x-normal.svg) - "~" 表示 _"x 的满足的分布是 ..."_

那么高斯分布（某些_x_可能是具有特定均值和方差的分布的一部分的概率）由下式给出：

![Gaussian Distribution](../../images/anomaly_detection/p.svg)

## 估计高斯分布的参数 Estimating Parameters for a Gaussian

我们可以使用以下公式来为第 _i<sup>th</sup>_ 个特征估计高斯参数(均值和方差)

![mu-i](../../images/anomaly_detection/mu-i.svg)

![sigma-i](../../images/anomaly_detection/sigma-i.svg)

![i](../../images/anomaly_detection/i.svg)

![m](../../images/anomaly_detection/m.svg) - number of training examples.

![n](../../images/anomaly_detection/n.svg) - number of features.

## 密度估计 Density Estimation

我们有以下的训练集合：

![Training Set](../../images/anomaly_detection/training-set.svg)

![x-in-R](../../images/anomaly_detection/x-in-R.svg)

我们假设训练集的每个特征都是符合正态分布的：

![x-1](../../images/anomaly_detection/x-1.svg)

![x-2](../../images/anomaly_detection/x-2.svg)

![x-n](../../images/anomaly_detection/x-n.svg)

那么:

![p-x](../../images/anomaly_detection/p-x.svg)

![p-x-2](../../images/anomaly_detection/p-x-2.svg)

## 异常检测算法 Anomaly Detection Algorithm

1. 从可能的异常例子(![Training Set](../../images/anomaly_detection/training-set.svg))中选择出特征![x-i](../../images/anomaly_detection/x-i.svg) .
2. 使用以下公式![mu-i](../../images/anomaly_detection/mu-i.svg)

![sigma-i](../../images/anomaly_detection/sigma-i.svg)拟合出参数 ![params](../../images/anomaly_detection/params.svg)

3. 给定新的样本 _x_, 计算出 _p(x)_:

![p-x-2](../../images/anomaly_detection/p-x-2.svg)

如果 ![anomaly](../../images/anomaly_detection/anomaly.svg) 则异常

![epsilon](../../images/anomaly_detection/epsilon.svg) - 概率阈值.

## 算法评估 Algorithm Evaluation

设计出的算法将通过 _F1 score_ 进行评估。

F1 score 是准确率和召回率的调和均值, 最理想的情况为 _1_ （最佳的准确率和召回率）,最坏的情况为 _0_。

![F1 Score](https://upload.wikimedia.org/wikipedia/commons/2/26/Precisionrecall.svg)

![f1](../../images/anomaly_detection/f1.svg)

其中:

![precision](../../images/anomaly_detection/precision.svg)

![recall](../../images/anomaly_detection/recall.svg)

_tp_ - 真阳性(true positives)的样本数量。

_fp_ - 假阳性(false positives)的样本数量。

_fn_ - 假阴性(false negatives)的样本数量。

## 参考文献 References

- [Machine Learning on Coursera](https://www.coursera.org/learn/machine-learning)
- [Normal Distribution on Wikipedia](https://en.wikipedia.org/wiki/Normal_distribution)
- [F1 Score on Wikipedia](https://en.wikipedia.org/wiki/F1_score)
- [Precision and Recall on Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall)
  