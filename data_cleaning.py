from collections import Counter
import pandas as pd


def read_action_file(path):
	"""分块读取给定的用户行为文件

	:param path: 文件路径
	:return: 读取的DataFrame
	"""
	reader = pd.read_csv(path, header=0, iterator=True)
	chunk_size = 1000000

	# 因文件过大，采用分块读取
	chunks = []
	while True:
		try:
			chunk = reader.get_chunk(chunk_size)[['user_id', 'sku_id', 'type', 'time']]
			chunks.append(chunk)
		except StopIteration:
			break

	# 将文件块拼接成DataFrame
	df_action = pd.concat(chunks, ignore_index=True)

	return df_action


def read_user_file(fname):
	"""分块读取给定的用户信息文件

	:param fname: 文件路径
	:return: 读取的DataFrame
	"""

	reader = pd.read_csv(fname, header=0, iterator=True, encoding='gbk')
	chunk_size = 1000000

	# 因文件过大，采用分块读取
	chunks = []
	while True:
		try:
			chunk = reader.get_chunk(chunk_size)[['user_id', 'age']]
			chunks.append(chunk)
		except StopIteration:
			break

	# 将文件块拼接成DataFrame
	df_action = pd.concat(chunks, ignore_index=True)

	return df_action


def action_count(group):
	"""对分组的行为进行统计

	根据传入的分组的所有行为信息，统计点击和购买行为的次数。

	:param group: 传入的分组的行为信息
	:return: 加上了行为统计信息的group
	"""
	behavior_type = group['type'].astype(int)
	type_counter = Counter(behavior_type)
	group['buy_num'] = type_counter[4]
	group['click_num'] = type_counter[6]

	return group[['user_id', 'sku_id', 'buy_num', 'click_num']]


def sku_analysis(df_action):
	"""对所有商品点击数进行统计

	:param df_action: 所有用户行为的DataFrame
	:return: 过滤掉访问量少的商品ID列表
	"""
	# 按照sku_id进行分组
	df_action = df_action.groupby('sku_id', as_index=False).apply(action_count)
	df_action = df_action.drop_duplicates('sku_id')

	# 过滤访问量少的商品
	df_action = df_action[df_action['click_num'] > 50]

	sku_list = list(df_action['sku_id'])

	return sku_list


def user_action_analysis(df_action):
	"""对用户的行为进行分析，去除惰性用户

	通过计算每个用户的购买转化率，将购买转化率过低的以及没有购买记录的用户进行剔除

	:param df_action: 所有用户行为的DataFrame
	:return: 剔除了惰性用户的用户list
	"""
	# 按照user_id进行分组统计
	df_action = df_action.groupby('user_id', as_index=False).apply(action_count)
	df_action = df_action.drop_duplicates('user_id')

	# 点击购买转化率
	df_action['buy_click_ratio'] = df_action['buy_num'] / df_action['click_num']

	# 去除惰性用户
	df_action = df_action[df_action['buy_click_ratio'] > 0.01]
	df_action = df_action[df_action['buy_click_ratio'] != float('inf')]

	# 去除没有点击记录的用户
	df_action = df_action[df_action['click_num'] != 0]

	user_list = list(df_action['user_id'])
	return user_list


def action_data_cleaning(fname):
	"""对用户行为数据表进行清洗，去除惰性用户和访问量过少商品的数据

	通过对用户行为数据进行分析，计算购买转化率和商品点击次数，去除惰性用户的行为数据的访问量过少商品数据

	:param fname: 清洗的行为数据文件路径
	:return: 清洗后的行为数据DataFrame
	"""
	df_action = read_action_file(fname)
	user_list = user_action_analysis(df_action)
	sku_list = sku_analysis(df_action)
	df_action = df_action[df_action['sku_id'].isin(sku_list)]
	df_action = df_action[df_action['user_id'].isin(user_list)]

	return df_action[['user_id', 'sku_id', 'type', 'time']]


def user_data_cleaning(fname):
	"""对用户信息表进行清洗，对年龄缺失值进行众数填充

	用数字替换年龄段，对缺失值进行众数填充，填充信息为：
	-1表示未知，0表示15岁以下，1表示16到25岁，2表示26到35岁，3表示36到45岁，4表示46到55岁，5表示56岁以上

	:param fname: 清洗的用户信息文件路径
	:return: 清洗后的用户信息DataFrame
	"""
	df_user = read_user_file(fname)
	# 将各年龄段的用户用数字代替
	df_user.replace({'15岁以下': 0, '16-25岁': 1, '26-35岁': 2, '36-45岁': 3, '46-55岁': 4, '56岁以上': 5}, inplace=True)
	# NaN用-1替换
	df_user.fillna(-1, inplace=True)
	df_user['age'] = df_user['age'].astype(int)
	# -1用众数替换
	df_user['age'].replace(-1, int(df_user['age'].mode()), inplace=True)

	return df_user


def data_cleaning(user_file, action_file):
	"""对行为数据和用户数据进行清洗

	对行为数据和用户数据进行清洗，并将两个DataFrame进行连接

	:param user_file: 用户信息文件路径
	:param action_file: 用户行为文件路径
	:return: 清洗并连接的DataFrame
	"""

	df_user = user_data_cleaning(user_file)
	df_action = action_data_cleaning(action_file)

	df = pd.merge(df_action, df_user, how='left', on='user_id')

	return df
