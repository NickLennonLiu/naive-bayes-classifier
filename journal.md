## 3.15
Created the basic framework. including:
- dataloader.py
- main.py
- model.py
- params.py

Where the Dataloader class can function to k-fold

Next up should be implementing the bag-of-words, as well as other possible features.

Found some reference:

https://blog.csdn.net/yzy_1996/article/details/104680315

https://scikit-learn.org/stable/modules/feature_extraction.html#stop-words

### Brainstorming

在jupyter里实验了一下不同类型邮件中的元信息的数量。

发现确实ham和spam的元信息数量确实有所不同。

但是我是否有必要专门地提取呢？难道这些元信息的名称不也在词袋的向量里了吗？

尝试了nltk的tokenize功能，发现它把一大堆东西都分开了...感觉不大行

还有怎么划分正文呢？

## 3.16

今天先专注于实现bayes本身的算法，再逐渐考虑特征吧。

好想从中搞出点demo看看自己有没有做对，但是好麻烦...

## 3.17
改了main中的一个小bug，目前看来是成功用词袋模型搞出了一个NBC了。

接下来要做的事情：
- 衡量训练和测试时间
- 完善流水线
- 开始整点花活

1. 现在我希望能够将结果输出到workdir中
2. 并且希望能规范一下输出