import cv2
import os
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
from net.inception import InceptionResNetV1

# 去掉tensorflow Cpu的报错
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class face_rec():
    # 得到仓库里所有人脸的128维度特征向量
    def __init__(self):
        # 创建mtcnn对象
        # 检测图片中的人脸
        self.mtcnn_model = mtcnn()
        # 门限函数
        self.threshold = [0.5,0.8,0.9]

        # 载入facenet的模型
        # 将检测到的人脸转化为128维的向量
        self.facenet_model = InceptionResNetV1()
        # model.summary()
        model_path = './model_data/facenet_keras.h5'
        self.facenet_model.load_weights(model_path)

        #-----------------------------------------------#
        #   对数据库中的人脸进行编码
        #   known_face_encodings中存储的是编码后的人脸
        #   known_face_names为人脸的名字
        #-----------------------------------------------#
        face_list = os.listdir("face_dataset")

        # 创建两个列表
        # 用于存放编码后的人脸
        self.known_face_encodings=[]
        # 用于存放编码后人脸的名字
        self.known_face_names=[]

        # 对face_dataset中的所有人脸进行遍历
        for face in face_list:
            # 把人脸读取出来
            name = face.split(".")[0]

            img = cv2.imread("./face_dataset/"+face)
            # 转成RGB，才可以用MTCNN进行识别
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            # 检测人脸
            rectangles = self.mtcnn_model.detectFace(img, self.threshold)

            # 转化成正方形
            rectangles = utils.rect2square(np.array(rectangles))
            # facenet要传入一个160x160的图片
            rectangle = rectangles[0]
            # 记下他们的landmark
            # 人脸上面的五个标记点，帮助进行人脸对齐。旋转45°依然能转正，检测到人脸。
            landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160
            # 把人脸截下来
            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img,(160,160))
            # 利用5个标记点进行对齐
            new_img,_ = utils.Alignment_1(crop_img,landmark)

            # 增加一个维度
            new_img = np.expand_dims(new_img,0)
            # 将检测到的人脸传入到facenet的模型中，实现128维特征向量的提取
            face_encoding = utils.calc_128_vec(self.facenet_model,new_img)

            # 把名字和特征向量存入已知
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)

    # 把读取出来的图片进行识别
    def recognize(self,draw):
        #-----------------------------------------------#
        #   人脸识别
        #   先定位，再进行数据库匹配
        #-----------------------------------------------#
        height,width,_ = np.shape(draw)
        draw_rgb = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)

        # 检测人脸
        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)

        if len(rectangles)==0:
            return

        # 转化成正方形
        rectangles = utils.rect2square(np.array(rectangles,dtype=np.int32))
        # 看看图片的高和宽是否超出范围，要把图片限制在范围内
        rectangles[:,0] = np.clip(rectangles[:,0],0,width)
        rectangles[:,1] = np.clip(rectangles[:,1],0,height)
        rectangles[:,2] = np.clip(rectangles[:,2],0,width)
        rectangles[:,3] = np.clip(rectangles[:,3],0,height)
        #-----------------------------------------------#
        #   对检测到的人脸进行编码
        #-----------------------------------------------#
        face_encodings = []
        for rectangle in rectangles:
            landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160
            # 将摄像头实时获取的图片进行截取
            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img,(160,160))
            # 并矫正该图片(对齐)
            new_img,_ = utils.Alignment_1(crop_img,landmark)
            new_img = np.expand_dims(new_img,0)
            # 传入facenet模型中，把128维向量保存在face_encodings列表中
            face_encoding = utils.calc_128_vec(self.facenet_model,new_img)
            face_encodings.append(face_encoding)

        face_names = []
        for face_encoding in face_encodings:
            # 取出一张脸并与数据库中所有的人脸进行对比，计算得分。实验得，一般不同人脸的差在1.4左右。
            # 门限tolerance为0.9,意思是当距离小于0.9时,实时监测到的人脸跟库中人脸有一定相似度，对比结果保存到matches中
            matches = utils.compare_faces(self.known_face_encodings, face_encoding, tolerance = 0.9)
            name = "Unknown"
            # 找出距离最近的人脸
            face_distances = utils.face_distance(self.known_face_encodings, face_encoding)
            # 取出这个最近人脸的评分
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            # 将找到的最接近的脸的名字保存到该列表
            face_names.append(name)

        # 取出方框的位置
        rectangles = rectangles[:,0:4]
        #-----------------------------------------------#
        #   画框，并写上名字
        #-----------------------------------------------#
        for (left, top, right, bottom), name in zip(rectangles, face_names):
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw, name, (left , bottom - 15), font, 0.75, (255, 255, 255), 2) 
        return draw

if __name__ == "__main__":

    dududu = face_rec()
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, draw = video_capture.read()
        dududu.recognize(draw) 
        cv2.imshow('Video', draw)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()