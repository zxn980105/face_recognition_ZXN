# 人脸截取与对齐矫正
import cv2
import numpy as np
from net.mtcnn import mtcnn
import utils.utils as utils
from net.inception import InceptionResNetV1


img = cv2.imread('face_dataset/timg.jpg')

# 创建mtcnn对象
mtcnn_model = mtcnn()
# 门限函数
threshold = [0.5,0.7,0.9]
# 检测人脸，rectangles中包含人脸位置和5个特征点的位置
rectangles = mtcnn_model.detectFace(img, threshold)

draw = img.copy()
# 转化成正方形，是为了方便facenet的处理
rectangles = utils.rect2square(np.array(rectangles))

# 载入facenet
facenet_model = InceptionResNetV1()
# model.summary()
model_path = './model_data/facenet_keras.h5'
facenet_model.load_weights(model_path)

# 因为例子只有1张人脸所以只循环了一次
for rectangle in rectangles:
    if rectangle is not None:
        # 把landmark的位置进行处理。注意，利用Mtcnn获取的位置参数都是相对左上角原点的，截取后的坐标原点会有变动，所以需要处理
        # 方法是先让标记点减去被截图像的左上角坐标，从而获得相对坐标
        # 然后除以人脸框的宽度，再乘以160（因为facenet的输入是160*160的向量）
        landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160
        # 截图这张图（y方向，x方向）
        crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]

        # 对齐方法1：利用双眼去矫正
        crop_img = cv2.resize(crop_img,(160,160))
        cv2.imshow("before",crop_img)
        new_img,_ = utils.Alignment_1(crop_img,landmark)
        cv2.imshow("two eyes",new_img)

        # 方法2：先在新图中指定好5个标记点的位置，再将原图的标记点与之重合并映射过去
        # std_landmark = np.array([[54.80897114,59.00365493],
        #                         [112.01078961,55.16622207],
        #                         [86.90572522,91.41657571],
        #                         [55.78746897,114.90062758],
        #                         [113.15320624,111.08135986]])
        # crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        # crop_img = cv2.resize(crop_img,(160,160))
        # new_img,_ = utils.Alignment_2(crop_img,std_landmark,landmark)
        # cv2.imshow("affine",new_img)

        new_img = np.expand_dims(new_img,0)
        feature1 = utils.calc_128_vec(facenet_model,new_img)

cv2.waitKey(0)