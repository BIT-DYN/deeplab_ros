# deeplab_ros

This is the ROS implementation of the semantic segmentation algorithm [Deeplab v3+](https://github.com/jfzhang95/pytorch-deeplab-xception).

We put two packages here for the convenience of using the correct version of Opencv.

## Install

### Environmental requirements
You need an Anaconda environment with [pytorch](https://pytorch.org/) installed.

In addition, a few third-party packages (such as _catkin_pkg visdom_) are required, but there is no specific statistics. 

You can install them when an error is reported.

### Clone

```bash
mkdir DeepLabV3Plus_ws && cd DeepLabV3Plus_ws
mkdir src && cd src
git clone git@github.com:BIT-DYN/deeplab_ros.git
cd ..
```
### Compile

```bash
source activate your_env
catkin config
catkin config -DPYTHON_EXECUTABLE=/home/user_name/anaconda3/envs/your_env/bin/python -DPYTHON_INCLUDE_DIR=/home/user_name/anaconda3/envs/your_env/include/python3.7m -DPYTHON_LIBRARY=/home/user_name/anaconda3/envs/your_env/lib/libpython3.7m.so -DCMAKE_BUILD_TYPE=Release -DSETUPTOOLS_DEB_LAYOUT=OFF
catkin_make
```

## Use

### Modify

The code needs to be modified for your own use. They are all in the file [Img_seg_ros/predict.py](https://github.com/BIT-DYN/deeplab_v3-_ros/blob/master/Img_seg_ros/predict.py).
Please modify your path or topic name.

1. Line 30
``` python
sys.path.append('/home/dyn/DeepLabV3Plus_ws/devel/lib/python3.6/site-packages')
```

2. Line 51
``` python
ckpt = "/home/dyn/DeepLabV3Plus_ws/src/deeplab_ros/Img_seg_ros/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
```

3. Line 112
``` python
img_top = "/miivii_gmsl_ros/camera0/image_raw"
```

### Run

```bash
source devel/setup.bash
roslaunch img_seg run.launch 
```
### Output
Open rviz to view the topic as '/image_ view/image_seg' color segmentation results.
The colors of various categories can be modified in the file [Img_seg_ros/datasets/cityscapes.py](The colors of various categories can be modified in the file.).
<div align=center>
<img src="https://github.com/BIT-DYN/deeplab_ros/blob/master/out.jpg" width="500">
</div>

## Other introductions
[CSDN](https://blog.csdn.net/weixin_43807148/article/details/123489519?spm=1001.2014.3001.5501)
