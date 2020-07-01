# -*-coding:utf-8-*-
"""
	author: k-starmon
	datetime: 2020/1/23
"""

def detect():
	"""
	function: 检测人脸并保存
	return: 人脸数量与时间	for:查找保存的图片
	"""
	import time
	import cv2 as cv
	import numpy as np

	faceCascade=cv.CascadeClassifier('D:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')
	#加载分类器
	cap = cv.VideoCapture(0)
	#调用一个摄像头
	while cap.isOpened():
		ret,frame = cap.read()
		#ret,bool表示图片有没有被读取，frame为返回的一帧的图片
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		#灰度化
		gray = cv.equalizeHist(gray)
		#均衡化
		faces = faceCascade.detectMultiScale(gray,1.3,5)
		#人脸检测，检测到五次才算
		if len(faces) > 0:
			#print(len(faces))
			num = 0
			day = time.strftime("%Y-%m-%d",time.localtime(time.time()))
			#获取当前日期：%Y四位数的当前年份，%m月，%d天
			for (x,y,w,h) in faces:
				frame = cv.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
				#识别出的人脸画框
				face = frame[y:y+h,x:x+w]
				num = num + 1
				cv.imwrite("./user_face/" + str(day) + "_" + str(num) + ".jpg", face)
				#单个人脸保存图片，包含日期
			cv.imshow('vedio',frame)
			if cv.waitKey(1) == ord('q'):
				break

			return len(faces), day
		else:
			return False, False
			print("未捕捉到人脸！")

	cap.release()
	cv.destroyAllWindows()


def recognition(sum,date):
	"""
	function: 人脸识别，从默认的保存路径中与用户的图片进行比较，若不是，则删除
	"""
	import os
	import face_recognition

	user_image = face_recognition.load_image_file("./user_face/user.jpg")
	#加载图片
	try:
		user_face_encoding = face_recognition.face_encodings(user_image)[0]
		#获取编码，列表，因为只有一个人脸，故索引[0]
		know_face = [user_face_encoding]
		#建在列表里，后面比较时维度相同
	except IndexError:
		print("固有图片加载失败!")
		quit()

	for i in range(1, sum+1):
		#遍历从1~sum
		unknown_face = face_recognition.load_image_file("./user_face/" + str(date) + "_" + str(i) +".jpg")
		try:
			unknown_face_encoding = face_recognition.face_encodings(unknown_face)[0]
		except IndexError:
			print("待识别图片加载失败!")
			quit()
		#result = face_recognition.compare_faces(know_face, unknown_face_encoding)
		#print(format(result[0]))
		#比较结果，为bool列表,和妹妹太像了，一直是true
		distance = face_recognition.face_distance(know_face, unknown_face_encoding)
		result = format(distance < 0.4)
		#变成向量的距离比较，调整到0.4，可以辨别和姐妹的差别，笑哭笑哭
		if result == "[False]":
			os.remove("./user_face/" + str(date) + "_" + str(i) +".jpg")
			print("不是用户图片")
			#若不是对象的图片，则删除保存
		else:
			print("用户已识别")


if __name__ == '__main__':
	"""
	人脸检测，识别，整个过程
	"""
	import os
	import json
	from cnn import CNN
	from transform import predict_expression
	#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	num, day = detect()
	#人脸检测
	if num != False:
		print("\n")
		print("检测到的人脸数：" + str(num) + "\t" + "日期：" + str(day))
		recognition(num, day)
		#人脸识别
		model = CNN()
		#加载cnn
		model.load_weights('./cnn3_best_weights.h5')
		#加载cnn的最佳权值
		for i in range(1, num+1):
			try:
				img = open("./user_face/" + str(day) + "_" + str(num) + ".jpg")
				emotion, number = predict_expression("./user_face/" + str(day) + "_" + str(num) + ".jpg", model)
				print("识别表情为：" + emotion)
				#print(number)
				result = [{"date": str(day)}, {"emotion": str(emotion)}]
				#result = [{"ID": str(ID)},{"date": str(day)}, {"emotion": str(emotion)}]
				with open("./result.json","w") as f:
					json.dump(result,f)
					print("结果已保存")
			except IOError:
				print(str(day) + "_" + str(num) + ".jpg" + "已被筛选删除！")
	else:
		print("没有检测到人脸")
		quit()