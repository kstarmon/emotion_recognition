emotion_recognition:将参考图片对待识别图片进行人脸识别，之后进行表情识别，并将结果写入数据库
表情识别参考https://github.com/luanshiyinyang/ExpressionRecognition。有问题联系k_starmon@bupt.edu,cn
write_mysql：读写数据库
detect_recognition.py：可调用摄像头

可识别表情:"anger",
	"disgust",
	"fear",
	"happy",
	"sad",
	"surprised",
	"neutral",
	"contempt"

python版本：3.6
调用库:	keras，numpy，os，face_recognition，opencv，datetime，json，pymysql，collections，PIL
	dlib(face_recognition依赖)，tensorflow(keras依赖)
使用注意：请先保存参考的照片在目录"./usr_face/",命名为用户名，格式为".jpg"。待识别图片在“./faces/”文件夹中，命名为年_月_日_时_随机数.jpg。

cnn3_best_weight.h5：cnn的最佳权值

API：month_analyse(user_name)
function：请求返回从今天起30天的心情
return：userName_month.json
说明：user_name代表用户名，字符串
例：
{
'user_name': 'name', 
"anger"：0.0,
"disgust"：0.1,
"fear"：0.1,
"happy"：0.8,
"sad"：0.0,
"surprised"：0.0,
"neutral"：0.0,
"contempt"：0.0
 }
API：day_analyse(user_name,date)
function：查看某一天的心情(一天内的情绪可能是变化的)
return：userName_day.json
说明：user_name,date格式为2020-1-1，字符串
例：
{
'user_name': 'name', 
"date":date
"anger"：0.0,
"disgust"：0.1,
"fear"：0.1,
"happy"：0.8,
"sad"：0.0,
"surprised"：0.0,
"neutral"：0.0,
"contempt"：0.0
}