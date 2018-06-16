# 项目说明
基于**YOLOv2 / Mask R-CNN**实现的视频蒙版弹幕**黑科技**，达到bilibili官方的[demo](https://www.bilibili.com/read/cv534194/)效果。    
> 注意，此项目仅仅是深度学习物体检测方面的一个实战课程demo，算是后端实现吧，并没有真正意义上在html5播放器中前端的实现（我也不会）。

# 预备课程推荐
- YOLO算法：[【中文】Yolo v1全面深度解读 目标检测论文](https://www.bilibili.com/video/av23354360)

- Mask R-CNN算法：[【中文】Mask R-CNN 深度解读与源码解析 目标检测](https://www.bilibili.com/video/av24795835)

# 本项目课程
- 蒙版弹幕实战:


# 视频效果截图
---------------------
- **Mask R-CNN模型下：**       
---------------

![image.png](https://upload-images.jianshu.io/upload_images/3251332-7781c53201c8b984.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![image.png](https://upload-images.jianshu.io/upload_images/3251332-8b783d254d225c8e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![image.png](https://upload-images.jianshu.io/upload_images/3251332-006a47b3fd2cc794.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![image.png](https://upload-images.jianshu.io/upload_images/3251332-54937dc30d554573.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![image.png](https://upload-images.jianshu.io/upload_images/3251332-812708a4d6d7c881.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---------------------
- **YOLOv2模型下**：    
---------------------

![image.png](https://upload-images.jianshu.io/upload_images/3251332-2572b92d438d20e8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![image.png](https://upload-images.jianshu.io/upload_images/3251332-18b39185626f0394.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



# Requirements

- numpy
- tensorflow
- keras
- opencv
- [darkflow](https://github.com/thtrieu/darkflow) ( YOLOv2的第三方实现）
- [Mask R-CNN](https://github.com/matterport/Mask_RCNN)
- ... ...


# 食用指南    

1. git clone [darkflow](https://github.com/thtrieu/darkflow)或者[Mask R-CNN](https://github.com/matterport/Mask_RCNN)开源库；
2. 确保开源库能顺利运行；
3. copy此库中的py文件到对应目录下；（你想用哪个模型就copy哪个文件夹）
4. 准备数据：一份是原始origin视频，另一份是带弹幕内嵌的danmu视频（可使用格式工厂压制弹幕）。
5. 修改py文件中相关配置与输入视频的文件名；
6. 等待输出……

