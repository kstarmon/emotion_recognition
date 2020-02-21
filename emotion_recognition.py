# -*-coding:utf-8-*-
"""
	author: Zhou Chen
	datetime: 2019/6/19 18:49
	upgrade: k-starmon
	datetime: 2020/1/23
	update: 2020/2/12 增添数据库写入
	last_update: 2020/2/21 增添人脸识别
	function：表情识别部分的所有 希望每日执行一次，如6:00
"""

def recognition(path):
	"""
	function: 人脸识别，从默认的保存路径中与用户的图片进行比较，若不是，则删除
	"""
	import os
	import face_recognition

	all_user = []
	for root, dirs, files in os.walk("./pre_face"):
	#遍历文件夹中的文件
		for file_name in files:
			file_name = file_name.split(".")
			all_user.append(file_name[0])

	know_face = []
	for i in range(0, len(all_user)):
		pre_path = "./pre_face/" + str(all_user[i]) + ".jpg"
		user_image = face_recognition.load_image_file(pre_path)
		#user_image = face_recognition.load_image_file("./user_face/user.png")
		#加载图片
		try:
			user_face_encoding = face_recognition.face_encodings(user_image)[0]
			#获取编码，列表，因为只有一个人脸，故索引[0]
			know_face.append(user_face_encoding)
			#建在列表里，后面比较时维度相同
		except IndexError:
			#print("固有图片加载失败!")
			quit()

	unknown_face = face_recognition.load_image_file(path)
	try:
		unknown_face_encoding = face_recognition.face_encodings(unknown_face)[0]
	except IndexError:
		#print("待识别图片加载失败!")
		quit()

	distance = face_recognition.face_distance(know_face, unknown_face_encoding)
	result = format(distance < 0.4)
	#变成向量的距离比较，调整到0.4

	n = len(result) - 1
	#以下为结果处理
	result = result[1:n]
	result = result.split(" ")
	result.remove("")
	if "True" in result:
		i = result.index("True")
		return str(all_user[i])
	else:
		os.remove(path)
		return "False"
		#print("不是用户图片")
		#若不是对象的图片，则删除保存


def face_detect(img_path):
	"""
	进一步检测图片的人脸，灰度化
	:param img_path: 图片的完整路径
	:return:
	"""
	import cv2

	face_cascade = cv2.CascadeClassifier('D:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')
	img = cv2.imread(img_path)

	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(
		img_gray,
		scaleFactor=1.1,
		minNeighbors=1,
		minSize=(30, 30)
	)
	return img, img_gray, faces


def generate_faces(face_img, img_size=48):
	"""
	将探测到的人脸进行增广
	:param face_img: 灰度化的单个人脸图
	:param img_size: 目标图片大小48*48
	:return:
	"""
	import cv2
	import numpy as np

	face_img = face_img / 255.
	face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
	resized_images = list()
	resized_images.append(face_img[:, :])
	resized_images.append(face_img[2:45, :])
	resized_images.append(cv2.flip(face_img[:, :], 1))
	#resized_images.append(cv2.flip(face_img[2], 1))
	#resized_images.append(cv2.flip(face_img[3], 1))
	#resized_images.append(cv2.flip(face_img[4], 1))
	resized_images.append(face_img[0:45, 0:45])
	resized_images.append(face_img[2:47, 0:45])
	resized_images.append(face_img[2:47, 2:47])

	for i in range(len(resized_images)):
		resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
		resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
	resized_images = np.array(resized_images)
	return resized_images


def index2emotion(index=0, kind='cn'):
    """
    根据表情下标返回表情字符串
    :param index:
    :return:
    """
    emotions = {
        '发怒': 'anger',
        '厌恶': 'disgust',
        '恐惧': 'fear',
        '开心': 'happy',
        '伤心': 'sad',
        '惊讶': 'surprised',
        '中性': 'neutral',
        '蔑视': 'contempt'

    }
    if kind == 'cn':
        return list(emotions.keys())[index]
    else:
        return list(emotions.values())[index]


def predict_expression(img_path, model):
	"""
	对图中n个人脸进行表情预测
	:param img_path:
	:return:
	"""
	import numpy as np
	import cv2

	border_color = (0, 0, 0)  
	#黑框框
	font_color = (255, 255, 255)  
	#白字字

	img, img_gray, faces = face_detect(img_path)
	if len(faces) == 0:
		return 'no', [0, 0, 0, 0, 0, 0, 0, 0]
	#遍历每一个脸
	emotions = []
	result_possibilitys = []
	for (x, y, w, h) in faces:
		face_img_gray = img_gray[y:y + h + 10, x:x + w + 10]
		faces_img_gray = generate_faces(face_img_gray)
		#预测结果线性加权
		results = model.predict(faces_img_gray)
		#利用加载的cnn进行预测
		result_sum = np.sum(results, axis=0).reshape(-1)
		label_index = np.argmax(result_sum, axis=0)
		emotion = index2emotion(label_index, 'en')
		#将结果转为对应的表情
		emotions.append(emotion)
		result_possibilitys.append(result_sum)
	return emotions[0], result_possibilitys[0]


def research_user(user):
	"""
	确认图片用户是否存在，可能有注销等情况
	return：bool,存在为True,不存在为false
	"""
	import pymysql

	db = pymysql.connect("localhost","guest","123456","win2020")
	#连接数据库，参数分别是本地连接，用户名，密码，数据库名
	cursor = db.cursor()
	#使用cursor()方法获取操作游标
	sql = "SELECT `user` FROM `user_info`"

	try:
		#执行sql语句
		cursor.execute(sql)
		#提交到数据库执行
		result = cursor.fetchall()
		db.commit()
		all_user = []
		for row in result:
			for id in row:
				all_user.append(id)
				#将返回的tuple变成列表形式
		if user in all_user:
			return True
			#print("该用户存在")
		else:
			return False
			#print("用户不存在！")
	except:
		db.rollback()
		#如果发生错误则回滚
		#print("未记录任何用户数据！")
		return False

def write_emotion(file_name, user_name, emotion):
	"""
	将结果写入数据库，可核对用户
	参数：图片名称，表情结果
	无返回
	"""
	import pymysql

	result = file_name.split("_")
	#文件名拆分
	user = user_name
	year = result[0]
	month = result[1]
	day = result[2]
	hour = result[3]
	date = str(year) + "-" + str(month) + "-"+ str(day)

	result = research_user(user)

	if result == True:

		db = pymysql.connect("localhost","guest","123456","win2020")
		#连接数据库，参数分别是本地连接，用户名，密码，数据库名
		cursor = db.cursor()
		#使用cursor()方法获取操作游标
		sql = "INSERT INTO `picture_emotion`(`user`, `date`, `hour`, `emotion`) VALUES (%s, %s, %s, %s)"

		try:
		#执行sql语句
			cursor.execute(sql, (user, date, hour, emotion))
			#提交到数据库执行
			#print(str(file_name) + "已导入！")
			db.commit()
		except:
			db.rollback()
			#print("未插入数据！")
			#如果发生错误则回滚
		db.close()
		#关闭数据库连接
	else:
		path = "./user_face/" + str(file_name)
		os.remove(path)
		#print("该用户不存在，图片已删除~")


def CNN(input_shape=(48, 48, 1), n_classes=8):
	"""
	参考论文实现
	A Compact Deep Learning Model for Robust Facial Expression Recognition
	:param input_shape:
	:param n_classes:
	:return:
	"""
	#input
	from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, AveragePooling2D
	from keras.models import Model
	from keras.layers.advanced_activations import PReLU
	import cv2 as cv
	import numpy as np
	from PIL import Image
	
	input_layer = Input(shape=input_shape)
	x = Conv2D(32, (1, 1), strides=1, padding='same', activation='relu')(input_layer)
	#block1
	x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
	x = PReLU()(x)
	x = Conv2D(64, (5, 5), strides=1, padding='same')(x)
	x = PReLU()(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
	#block2
	x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
	x = PReLU()(x)
	x = Conv2D(64, (5, 5), strides=1, padding='same')(x)
	x = PReLU()(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
	#fc
	x = Flatten()(x)
	x = Dense(2048, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(1024, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(n_classes, activation='softmax')(x)

	model = Model(inputs=input_layer, outputs=x)

	return model


if __name__ == '__main__':
	"""
	测试
	"""
	import os
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	model = CNN()
	model.load_weights('./cnn3_best_weights.h5')
	#加载权值

	for root, dirs, files in os.walk(".//user_face"):
	#遍历文件夹中的文件
		for file_name in files:
			path = "./user_face/" + str(file_name)

			result = recognition(path)
			if result != "False":
				emotion, number = predict_expression(path, model)
				#表情识别
				write_emotion(file_name, result, emotion)
				#写入数据库

