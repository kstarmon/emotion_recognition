# -*-coding:utf-8-*-
"""
	author: k-starmon
	datetime: 2020/2/9
	update：2020/2/15
	function：包括新用户信息写入数据库，用户注销，计算30天或者某一天的情绪结果写成.json文件
"""
def add_user(id,pwd):
	"""
	function：将新用户的个人信息保存，写入数据库
	参数:id用户名，pwd密码
	"""
	import pymysql

	user = str(id)
	pwd = str(pwd)

	db = pymysql.connect("localhost","guest","123456","win2020")
	#连接数据库，参数分别是本地连接，用户名，密码，数据库名
	cursor = db.cursor()
	#使用cursor()方法获取操作游标
	sql = "INSERT INTO `user_info`(`user`, `pwd`) VALUES (%s, %s)"
	
	try:
	#执行sql语句
		cursor.execute(sql, (user, pwd))
		#提交到数据库执行
		db.commit()
		#print("用户已导入")
	except:
		db.rollback()
		#print("未插入数据！")
		#如果发生错误则回滚

	db.close()
	#关闭数据库连接


def delete_user(id):
	"""
	function：将注销用户的个人信息删除,需要操作所有数据表
	参数:id用户名，pwd密码
	"""
	import pymysql

	id = str(id)

	db = pymysql.connect("localhost","guest","123456","win2020")
	#连接数据库，参数分别是本地连接，用户名，密码，数据库名
	cursor = db.cursor()
	#使用cursor()方法获取操作游标
	sql_emotion = "DELETE FROM `picture_emotion` WHERE `user` = %s"
	sql_info = "DELETE FROM `user_info` WHERE `user` = %s"

	try:
	#执行sql语句
		cursor.execute(sql_emotion, (id))
		#提交到数据库执行
		#print("用户表情数据已删除")
		cursor.execute(sql_info, (id))
		#print("用户基本数据已删除")
		db.commit()
	except:
		db.rollback()
		#如果发生错误则回滚

	db.close()
	#关闭数据库连接


def day_analyse(user, date):
	"""
	function：返回某一天的各个表情比重
	"""
	import json
	import pymysql
	from collections import Counter

	user = str(user)
	date = str(date)

	db = pymysql.connect("localhost","guest","123456","win2020")
	#连接数据库，参数分别是本地连接，用户名，密码，数据库名
	cursor = db.cursor()
	#使用cursor()方法获取操作游标
	sql = "SELECT `emotion` FROM `picture_emotion` WHERE `user` = %s and `date`= %s"

	try:
	#执行sql语句
		cursor.execute(sql, (user, date))
		#提交到数据库执行
		result = cursor.fetchall()
		db.commit()

		emotions = []
		for row in result:
			for emotion in row:
				emotions.append(emotion)
				#将返回的tuple变成列表形式

		num = len(emotions)
		#一天内记录的表情总个数和各个表情的个数
		#Counter(emotions)
		sad = emotions.count('sad')
		fear = emotions.count('fear')
		happy = emotions.count('happy')
		anger = emotions.count('anger')
		disgust = emotions.count('disgust')
		neutral = emotions.count('neutral')
		contempt = emotions.count('contempt')
		surprised = emotions.count('surprised')

		day_result = {
					'user':str(user), 'date':str(date),'anger':anger/num, 'disgust':disgust/num,
					'fear':fear/num, 'happy':happy/num, 'sad':sad/num, 
					'surprised':surprised/num,'neutral':neutral/num, 'contempt':contempt/num
					}
		day_result = json.dumps(day_result, indent=4)
		with open(str(user) + '_day.json', 'w') as f:
			f.write(day_result)
	except:
		db.rollback()
		#如果发生错误则回滚
		#print("查询错误！")

	db.close()
	#关闭数据库连接

def month_analyse(user):
	"""
	function：返回从今天倒数共30天的记录结果
	"""
	import json
	import pymysql
	import datetime

	user = str(user)
	today = datetime.date.today()
	pre_date = today - datetime.timedelta(days = 30)
	#获取今天日期和30天前的日期

	db = pymysql.connect("localhost","guest","123456","win2020")
	#连接数据库，参数分别是本地连接，用户名，密码，数据库名
	cursor = db.cursor()
	#使用cursor()方法获取操作游标
	sql = "SELECT `emotion` FROM `picture_emotion` WHERE `user` = %s and (`date`>= %s and `date`<= %s)"

	try:
	#执行sql语句
		cursor.execute(sql, (user, pre_date, today))
		#提交到数据库执行
		result = cursor.fetchall()
		#print(result)
		#print(type(result))
		db.commit()

		emotions = []
		for row in result:
			for emotion in row:
				emotions.append(emotion)
				#将返回的tuple变成列表形式

		#print(emotions)

		num = len(emotions)
		#一天内记录的表情总个数和各个表情的个数
		#Counter(emotions)
		#print(num)
		sad = emotions.count('sad')
		fear = emotions.count('fear')
		happy = emotions.count('happy')
		anger = emotions.count('anger')
		disgust = emotions.count('disgust')
		neutral = emotions.count('neutral')
		contempt = emotions.count('contempt')
		surprised = emotions.count('surprised')

		month_result = {
					'user':str(user), 'anger':anger/num, 'disgust':disgust/num,
					'fear':fear/num, 'happy':happy/num, 'sad':sad/num, 
					'surprised':surprised/num,'neutral':neutral/num, 'contempt':contempt/num
					}
		month_result = json.dumps(month_result, indent=4)
		with open(str(user) + '_month.json', 'w') as f:
			f.write(month_result)
	except:
		db.rollback()
		#如果发生错误则回滚
		#print("查询错误！")

	db.close()
	#关闭数据库连接

