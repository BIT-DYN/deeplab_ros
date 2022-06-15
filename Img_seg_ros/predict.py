#!/usr/bin/env python3
print('\n*****************************************\n\t[test libraries]:\n')

from pyexpat import model
import network
import utils
import os
import numpy as np

from datasets import  Cityscapes
from torchvision import transforms as T

import torch
import torch.nn as nn

from PIL import Image as im

import sys
import cv2
print(' - cv2.__file__ = ',cv2.__file__)

import rospy

from sensor_msgs.msg import Image
from std_msgs.msg import Header
import sensor_msgs
print(' - sensor_msgs.__file__ = ',sensor_msgs.__file__)
# 为了保证从我的编译文件家在模型，这块要自己改代码
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
sys.path.append('/home/dyn/DeepLabV3Plus_ws/devel/lib/python3.6/site-packages')
from cv_bridge import CvBridge, CvBridgeError
import cv_bridge
print(' - cv_bridge.__file__ = ',cv_bridge.__file__)
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')

print('\n*****************************************\n\t[finish test]\n')


# 加载模型文件，并设置为全局可访问的变量
def load_model():
    rospy.init_node('img_subscriber',  anonymous=True)
    global model, device

    # 选择GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    # 模型选了一个最好的
    model = network.deeplabv3plus_mobilenet(num_classes=19, output_stride=16)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # 从路径加载模型
    ckpt = "/home/dyn/DeepLabV3Plus_ws/src/deeplab_v3-_ros/Img_seg_ros/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
    if os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)
    model = model.eval()
    


def ImgCallback(data):
    global bridge, device
    try:
        cv_img = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
        return
    # print('receive a img')
    # 用opencv查看图像
    # cv2.imshow("cv_img" , cv_img)
    # cv2.waitKey(3)

    # 对图像进行语义分割
    # 先转化为PIL
    pil_img = im.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    # 图像输入时的参数
    transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    with torch.no_grad():
        pil_img = transform(pil_img).unsqueeze(0) # To tensor of NCHW
        pil_img = pil_img.to(device)
            
        pred = model(pil_img).max(1)[1].cpu().numpy()[0] # HW
        decode_fn = Cityscapes.decode_target
        colorized_preds = decode_fn(pred).astype('uint8')
        colorized_preds = im.fromarray(colorized_preds)
        seg_cv_img = cv2.cvtColor(np.asarray(colorized_preds),cv2.COLOR_RGB2BGR)

        # 用opencv查看图像
        # cv2.imshow("seg_cv_img" , seg_cv_img)
        # cv2.waitKey(3)

        # 发布语义图像到ROS
        ros_frame = bridge.cv2_to_imgmsg(seg_cv_img, "bgr8")
        ros_frame.header.stamp = rospy.Time.now()
        image_pub.publish(ros_frame) #发布消息


def img_subscriber():
    # img_top = "/image_view/image_raw"
    img_top = "/miivii_gmsl_ros/camera0/image_raw"
    global bridge, image_pub
    bridge = CvBridge()
    image_pub = rospy.Publisher('/image_view/image_seg',Image,queue_size = 1) #定义话题
    rospy.Subscriber(img_top, Image, ImgCallback)
    rospy.spin()


if __name__ == '__main__':
    load_model()
    img_subscriber()
