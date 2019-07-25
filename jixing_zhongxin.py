from collections import  deque
import numpy as np
#import imutils
import cv2
import time
#设定蓝色阈值，HSV空间
blueLower = np.array([100, 100, 100])
blueUpper = np.array([120, 255, 255])
#初始化追踪点的列表
mybuffer = 64
pts = deque(maxlen=mybuffer)
#打开摄像头
camera = cv2.VideoCapture(0)
#等待两秒
time.sleep(2)
#遍历每一帧，检测蓝色瓶盖
while True:
    #读取帧
    (ret, frame) = camera.read()
    frame=cv2.flip(frame, 1, dst=None)  # 水平镜像
    #判断是否成功打开摄像头
    if not ret:
        print ('No Camera')
        break
    #frame = imutils.resize(frame, width=600)
    #转到HSV空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #根据阈值构建掩膜
    mask = cv2.inRange(hsv, blueLower, blueUpper)
    #腐蚀操作
    mask = cv2.erode(mask, None, iterations=2)
    #膨胀操作，其实先腐蚀再膨胀的效果是开运算，去除噪点
    mask = cv2.dilate(mask, None, iterations=2)
    #轮廓检测
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    #初始化瓶盖圆形轮廓质心
    center = None
    #如果存在轮廓
    if len(cnts) > 0:
        #找到面积最大的轮廓
        c = max(cnts, key = cv2.contourArea)
        #确定面积最大的轮廓的矩形
        x, y, w, h = cv2.boundingRect(c)
        #计算轮廓的矩
        M = cv2.moments(c)
        #计算质心
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
        #只有当半径大于10时，才执行画图
        print(center)
        cv2.rectangle(frame, (x,y),(x+w, y+h), (0, 0, 255), 2)#显示矩形框
        cv2.circle(frame, center, 1, (0, 0, 255), -1)#显示圆心
    #转到HSV空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #根据阈值构建掩膜
    mask = cv2.inRange(hsv, blueLower, blueUpper)

    cv2.imshow('Frame', frame)
    #键盘检测，检测到esc键退出
    k = cv2.waitKey(5)&0xFF
    if k == 27:
        break
#摄像头释放
camera.release()
#销毁所有窗口
cv2.destroyAllWindows()
