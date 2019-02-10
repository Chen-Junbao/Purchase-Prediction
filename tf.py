import tensorflow as tf

from get_data import Data

# 超参数
lstm_1_time_step = 100  # 时间步
lstm_1_num_hidden = 128  # 隐藏层神经元数量
lstm_1_output_num = 100  # 第一层LSTM输出神经元数量

lstm_2_time_step = 20
lstm_2_num_hidden = 128
lstm_2_output_num = 20  # 第二层LSTM输出神经元数量

# 获取训练集和测试集
data = Data(lstm_1_time_step, lstm_2_time_step)
x_train_browse, x_train_cart, y_train, x_test_browse, x_test_cart, y_test, age_train, age_test = data.get_data_set()
x_train_browse = (x_train_browse - x_train_browse.min()) / (x_train_browse.max() - x_train_browse.min())
x_train_cart = (x_train_cart - x_train_cart.min()) / (x_train_cart.max() - x_train_cart.min())
x_test_browse = (x_test_browse - x_test_browse.min()) / (x_test_browse.max() - x_test_browse.min())
x_test_cart = (x_test_cart - x_test_cart.min()) / (x_test_cart.max() - x_test_cart.min())

"""------第一层LSTM结构------"""
with tf.name_scope('lstm_1'):
	lstm_1_input = tf.placeholder(dtype=tf.float32, shape=(None, lstm_1_time_step, 1), name='lstm_1_input')

	lstm_1_weight = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[lstm_1_num_hidden, lstm_1_output_num]),
	                            name='lstm_1_weight')
	lstm_1_bias = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[lstm_1_output_num]), name='lstm_1_bias')

	lstm_1_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_1_num_hidden)

	# 第一层LSTM输出
	with tf.variable_scope('lstm1'):
		outputs_1, states_1 = tf.nn.dynamic_rnn(lstm_1_cell, lstm_1_input, dtype=tf.float32)
		o_1 = tf.matmul(outputs_1[:, -1, :], lstm_1_weight) + lstm_1_bias

"""------第二层LSTM结构------"""
with tf.name_scope('lstm_2'):
	lstm_2_input = tf.placeholder(dtype=tf.float32, shape=(None, lstm_2_time_step, 1), name='lstm_2_input')

	lstm_2_weight = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[lstm_2_num_hidden, lstm_2_output_num]),
	                            name='lstm_2_weight')
	lstm_2_bias = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[lstm_2_output_num]), name='lstm_2_bias')

	lstm_2_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_2_num_hidden)

	# 第二层LSTM输出
	with tf.variable_scope('lstm2'):
		outputs_2, states_2 = tf.nn.dynamic_rnn(lstm_2_cell, lstm_2_input, dtype=tf.float32)
		o_2 = tf.matmul(outputs_2[:, -1, :], lstm_2_weight) + lstm_2_bias

"""------CNN结构------"""
with tf.name_scope('cnn'):
	cnn_input = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='cnn_input')
	x = tf.concat([cnn_input, o_1, o_2], axis=1, name='x')
	x = tf.reshape(tensor=x, shape=[-1, 11, 11, 1])
	y = tf.placeholder(dtype=tf.float32, shape=(None, data.sku_all_num), name='y')

	# 第一个卷积层
	W1 = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[3, 3, 1, 16], stddev=0.05))
	B1 = tf.Variable(tf.constant(0.5))

	conv1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME') + B1)
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	drop1 = tf.nn.dropout(pool1, keep_prob=0.25)

	# 第二个卷积层
	W2 = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[3, 3, 16, 32], stddev=0.05))
	B2 = tf.Variable(tf.constant(0.5))

	conv2 = tf.nn.relu(tf.nn.conv2d(drop1, W2, strides=[1, 1, 1, 1], padding='SAME') + B2)
	pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	drop2 = tf.nn.dropout(pool2, keep_prob=0.25)

	# 第三个卷积层
	W3 = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[3, 3, 32, 32], stddev=0.05))
	B3 = tf.Variable(tf.constant(0.5))

	conv3 = tf.nn.relu(tf.nn.conv2d(drop2, W3, strides=[1, 1, 1, 1], padding='SAME') + B3)
	pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	drop3 = tf.nn.dropout(pool3, keep_prob=0.25)

	# 全连接层
	full_connection_layer_size = int(pool3.shape[1] * pool3.shape[2] * pool3.shape[3])
	Wf = tf.Variable(tf.random_normal(shape=[full_connection_layer_size, 1024]))
	Bf = tf.Variable(tf.constant(0.5))
	pool3 = tf.reshape(pool3, (-1, full_connection_layer_size))
	full = tf.nn.relu(tf.matmul(pool3, Wf) + Bf)
	drop_full = tf.nn.dropout(full, keep_prob=0.5)

	# 输出层
	Wo = tf.Variable(tf.random_normal(shape=[1024, data.sku_all_num]))
	Bo = tf.Variable(tf.constant(0.5))
	o = tf.matmul(drop_full, Wo) + Bo
	prob_o = tf.nn.softmax(logits=o, axis=1, name='predict')

	# Loss
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=o, labels=y), name='loss')
	tf.summary.scalar('loss', cross_entropy)

	# Optimizer
	opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

	# Accuracy
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prob_o, 1), tf.argmax(y, 1)), 'float'))
	tf.summary.scalar('accuracy', accuracy)

	top_5 = tf.keras.metrics.top_k_categorical_accuracy(y_true=y, y_pred=o, k=5)
	tf.summary.scalar('top-5', top_5)

"""------Train Model------"""
epoch = 1000
batch_size = 50
batch_num = y_train.shape[0] // batch_size

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
	sess.run(tf.global_variables_initializer())

	# Summary
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter('train/', sess.graph)

	for i in range(epoch):
		for j in range(batch_num):
			loss, _, summary = sess.run([cross_entropy, opt, merged],
			                            feed_dict={lstm_1_input: x_train_browse[j * batch_size: (j + 1) * batch_size],
			                                       lstm_2_input: x_train_cart[j * batch_size: (j + 1) * batch_size],
			                                       cnn_input: age_train[j * batch_size: (j + 1) * batch_size],
			                                       y: y_train[j * batch_size: (j + 1) * batch_size]})
			writer.add_summary(summary, i * batch_num + j)
			if j % 10 == 0:
				print('epoch %d, batch %d / %d---loss: %f' % (i, j, batch_num, loss))
		accurate_rate = sess.run(accuracy,
		                         feed_dict={lstm_1_input: x_test_browse, lstm_2_input: x_test_cart, cnn_input: age_test,
		                                    y: y_test})
		top_5_acc = sess.run(top_5,
		                     feed_dict={lstm_1_input: x_test_browse, lstm_2_input: x_test_cart, cnn_input: age_test,
		                                y: y_test})
		print('accuracy ', accurate_rate)
		print('top-5 ', top_5_acc)

	tf.train.Saver().save(sess, 'checkpoint/ckp')
