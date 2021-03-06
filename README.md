# 用户购买商品预测
此项目采用LSTM以及CNN对用户的购买行为进行预测。通过两层LSTM网络分别对用户的浏览行为以及加购行为进行建模，两层LSTM网络的输出结合用户年龄特征作为CNN网络的输入，最后通过CNN网络给出用户购买的商品ID。
<br />
<br />
## 数据集
数据集采用京东JData竞赛中高潜用户购买意向预测竞赛给定的数据集。以下为经过数据清洗后的数据集：
<br />
+ 用户信息表<br />

| 用户ID | 用户年龄 | 用户性别 |
| ------ | ------- | ------- |
| 标记每个用户 | 清洗后由0,1,2,3,4,5表示 | 男，女以及保密 |

+ 商品信息表<br />

| 商品ID |
| ------ |
| 标记每个商品 |

+ 行为信息表<br />

| 用户ID | 商品ID | 行为时间 | 行为类型 |
| ------ | ------- | ------- | ------ |
| 标记每个用户 | 标记每个商品 | 产生行为对应时间 | 浏览，加购以及下单 |

## 神经网络结构
此项目采用LSTM和CNN结合的神经网络结构。通过LSTM网络对用户时序特征进行建模。第一层LSTM网络的输入为用户浏览商品的顺序，time_step为100；第二层LSTM网络的输入为用户加购商品的顺序，time_step为20；两层LSTM的输出加入用户自身年龄特征作为用户画像，shape为11*11，以此作为CNN网络的输入。CNN网络的输出为对于所有商品，用户购买的预测情况。

## 实验结果
经过100迭代之后，Top-1预测正确率为1%，Top-5预测正确率为5%。
