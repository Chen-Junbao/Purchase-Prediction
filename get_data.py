import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences


class Data:
	time_step = 0  # 时间步
	sku_all_num = 0  # 所有商品数量

	def __init__(self, time_step):
		self.time_step = time_step

	@staticmethod
	def read_data(path):
		""" 根据清洗后的csv文件路径获取DataFrame

		读取清洗后的csv文件，并删除重复行。之后选取含有浏览记录以及购买记录的数据，返回对应的DataFrame。
		:param path: 清洗后的csv文件路径
		:return: 对应数据的DataFrame
		"""
		df = pd.read_csv(path).drop_duplicates()
		df = df[(df['type'] == 4) | (df['type'] == 1)]
		df.sort_values(['user_id', 'time'], inplace=True)

		return df

	@staticmethod
	def generate_data_set(df, sku_all):
		""" 根据对应的DataFrame生成相应的样本及标签。

		将用户的浏览记录作为样本，浏览之后的购买商品ID作为标签。标签为1的下标在sku_all列表中对应的商品ID为购买的商品ID。即若label[2]=1，则购买的商品ID为sku_all[2]
		:param df: 给定的DataFrame
		:param sku_all: 所有的商品ID列表
		:return: 样本及标签数组
		"""
		groups = df.groupby('user_id')

		sample = []
		label = []

		for group in groups:
			type_list = list(group[1]['type'])
			sku_list = list(group[1]['sku_id'])
			start = 0
			for i in range(len(type_list)):
				if type_list[i] == 4:
					sample.append(np.array(sku_list[start: i]))

					temp = np.zeros(len(sku_all))
					temp[sku_all.index(sku_list[i])] = 1
					label.append(temp)

					start = i + 1

		return np.array(sample), np.array(label)

	@staticmethod
	def fit_sample(sample, time_step):
		""" 对样本集进行处理，使其最大长度为time_step。

		判断样本的长度，若样本长度大于time_step，则取最后的time_step个作为样本。
		:param sample: 需要处理的样本集
		:param time_step: 时间步
		:return: 处理后的样本集
		"""
		for i in range(len(sample)):
			if len(sample[i]) > time_step:
				sample[i] = sample[i][-time_step:]

		return sample

	def get_data_set(self):
		"""
		获取训练集和测试集
		:return: 训练样本，训练标签，测试样本，测试标签
		"""
		# 读取清洗后的csv文件
		df_train = self.read_data('train.csv')
		df_test = self.read_data('test.csv')

		# 训练集和测试集所包含的所有商品ID
		sku_all = list(set(df_train['sku_id']).union(set(df_test['sku_id'])))
		self.sku_all_num = len(sku_all)

		# 为训练集和测试集生成对应的样本及标签
		x_train, y_train = self.generate_data_set(df_train, sku_all)
		x_test, y_test = self.generate_data_set(df_test, sku_all)

		# 对样本集进行处理
		x_train = self.fit_sample(x_train, self.time_step)
		x_test = self.fit_sample(x_test, self.time_step)

		# 对小于time_step的样本进行扩充
		x_train = pad_sequences(sequences=x_train, maxlen=self.time_step, padding='pre')
		x_test = pad_sequences(sequences=x_test, maxlen=self.time_step, padding='pre')

		x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
		x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

		return x_train, y_train, x_test, y_test
