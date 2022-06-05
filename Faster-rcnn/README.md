#计算机视觉期末project

##训练步骤
###1、数据集的准备
本文使用VOC格式进行训练，训练前需要下载好VOC07的数据集，解压后放在根目录
将训练集和测试集分别放在VOCdevkit和VOVCdevkit-test下，结构如下：\
训练集：\
VOCdevkit\
VOC2007\
 &emsp;&emsp;Annotations\
&emsp;&emsp;&emsp;&emsp;ImageSets\
&emsp;&emsp;&emsp;&emsp;JPEGImages\
&emsp;&emsp;&emsp;&emsp;SegmentationClass\
&emsp;&emsp;&emsp;&emsp;SegmentationObject\
测试集:\
VOCdevkit-test\
VOC2007\
 &emsp;&emsp;Annotations\
&emsp;&emsp;&emsp;&emsp;ImageSets\
&emsp;&emsp;&emsp;&emsp;JPEGImages\
&emsp;&emsp;&emsp;&emsp;SegmentationClass\
&emsp;&emsp;&emsp;&emsp;SegmentationObject

之后分别运行voc_annotation.py和voc_annotation_test.py文件生成2007_train.txt、2007_val.txt、2007_test.txt.\
###2、开始网络训练
train.py的默认参数用于训练VOC数据集，直接运行train.py即可开始训练。训练过程中会将训练集、验证集、测试集的损失以及最好的权重信息存到logs文件夹内
###3、三个任务的实现

- 随机初始化训练VOC\
将train.py里的pretrained设置为False即可
- ImageNet预训练backbone网络，然后使用VOC进行fine tune\
将train.py里的pretrained设置为True即可,按照resnet50.py的resnet50函数的链接改为https://download.pytorch.org/models/resnet50-19c8e357.pth 即可。
- 使用coco训练的Mask R-CNN的backbone网络参数，初始化Faster R-CNN的backbone网络，然后使用VOC进行fine tune\
将train.py里的pretrained设置为True即可,按照resnet50.py的resnet50函数的链接改为https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth 即可。
###权重及损失链接：
https://drive.google.com/drive/folders/1mbnbYHFK182W0bl2BiWrif9uoSvAubNj?usp=sharing 

