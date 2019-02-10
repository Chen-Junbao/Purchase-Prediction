import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences


class Data:
	time_step_1 = 0  # LSTM1的时间步
	time_step_2 = 0  # LSTM2的时间步
	sku_all_num = 0  # 所有商品数量

	def __init__(self, time_step_1, time_step_2):
		self.time_step_1 = time_step_1
		self.time_step_2 = time_step_2

	@staticmethod
	def read_data(path):
		""" 根据清洗后的浏览记录csv文件路径获取DataFrame

		读取清洗后的csv文件，并删除重复行。之后选取含有浏览记录，加购记录以及购买记录的数据，返回对应的DataFrame。
		:param path: 清洗后的csv文件路径
		:return: 对应数据的DataFrame
		"""
		df = pd.read_csv(path).drop_duplicates()

		df_browse = df[(df['type'] == 4) | (df['type'] == 1)]
		df_browse = df_browse.sort_values(['user_id', 'time'])

		df_cart = df[(df['type'] == 4) | (df['type'] == 2)]
		df_cart = df_cart.sort_values(['user_id', 'time'])

		return df_browse, df_cart

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
					index = sku_all.index(sku_list[i])
					temp[index] = 1
					label.append(temp)

					start = i + 1

		return np.array(sample), np.array(label)

	@staticmethod
	def generate_age_set(df):
		""" 根据对应的DataFrame生成相应的样本及标签。

		将用户的浏览记录作为样本，浏览之后的购买商品ID作为标签。标签为1的下标在sku_all列表中对应的商品ID为购买的商品ID。即若label[2]=1，则购买的商品ID为sku_all[2]
		:param df: 给定的DataFrame
		:return: 样本及标签数组
		"""
		groups = df.groupby('user_id')

		sample = []

		for group in groups:
			type_list = list(group[1]['type'])
			for i in range(len(type_list)):
				if type_list[i] == 4:
					sample.append(group[1]['age'].iloc[0])

		sample = np.array(sample)
		sample = np.reshape(sample, (-1, 1))

		return sample

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
		df_browse_train, df_cart_train = self.read_data('train.csv')
		df_browse_test, df_cart_test = self.read_data('test.csv')

		# 训练集和测试集所包含的所有商品ID
		sku_all = list(set(df_browse_train['sku_id']).union(set(df_browse_test['sku_id'])))
		self.sku_all_num = len(sku_all)

		# 为训练集和测试集生成对应的样本及标签
		x_train_browse, y_train = self.generate_data_set(df_browse_train, sku_all)
		x_train_cart, _ = self.generate_data_set(df_cart_train, sku_all)
		x_test_browse, y_test = self.generate_data_set(df_browse_test, sku_all)
		x_test_cart, _ = self.generate_data_set(df_cart_test, sku_all)

		# 对样本集进行处理
		x_train_browse = self.fit_sample(x_train_browse, self.time_step_1)
		x_test_browse = self.fit_sample(x_test_browse, self.time_step_1)
		x_train_cart = self.fit_sample(x_train_cart, self.time_step_2)
		x_test_cart = self.fit_sample(x_test_cart, self.time_step_2)

		# 对小于time_step的样本进行扩充
		x_train_browse = pad_sequences(sequences=x_train_browse, maxlen=self.time_step_1, padding='pre')
		x_test_browse = pad_sequences(sequences=x_test_browse, maxlen=self.time_step_1, padding='pre')
		x_train_cart = pad_sequences(sequences=x_train_cart, maxlen=self.time_step_2, padding='pre')
		x_test_cart = pad_sequences(sequences=x_test_cart, maxlen=self.time_step_2, padding='pre')

		x_train_browse = x_train_browse.reshape((x_train_browse.shape[0], x_train_browse.shape[1], 1))
		x_test_browse = x_test_browse.reshape((x_test_browse.shape[0], x_test_browse.shape[1], 1))
		x_train_cart = x_train_cart.reshape((x_train_cart.shape[0], x_train_cart.shape[1], 1))
		x_test_cart = x_test_cart.reshape((x_test_cart.shape[0], x_test_cart.shape[1], 1))

		age_train = self.generate_age_set(df_browse_train)
		age_test = self.generate_age_set(df_browse_test)

		return x_train_browse, x_train_cart, y_train, x_test_browse, x_test_cart, y_test, age_train, age_test
