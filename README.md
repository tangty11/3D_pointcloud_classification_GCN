# 基于GCN的点云分类

#### 任务：
使用多层GCN网络实现三维点云的分类。

#### 注意:
1. 请自己写代码建图等，请不要直接套用已有代码
2. 可以自己做些设计和ablation studies，看看不同design的效果

#### GCN的实现：
PyTorch：https://github.com/tkipf/pygcn

#### 参考文献：
[1] T. N. Kipf, and M. Welling, “Semi-Supervised Classification with Graph Convolutional Networks,” in 5th International Conference on Learning Representations, 2017.
[2] R. Q. Charles, H. Su, M. Kaichun, and L. J. Guibas, “PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 77-85.

#### 数据集: 
ModelNet40

给的网址 (https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)打不开，在飞桨平台下： https://aistudio.baidu.com/datasetdetail/35331

#### 数据集读取方式：
https://github.com/WangYueFt/dgcnn/blob/master/pytorch/data.py



# run

权重文件在runs/，可用tensorboard查看loss、acc

#### GCN
python train_acc.py --exp_name=gcn_acc --model_name=gcn_acc --epochs=60

#### DGCNN
python train_acc.py --exp_name=dgcnn_acc --model_name=dgcnn_acc --epochs=60

#### PointNet
python train_acc.py --exp_name=pointnet_acc --model_name=pointnet_acc --epochs=60

#### ResidualGCN
python train_acc.py --exp_name=gcnresnet_acc --model_name=gcnresnet_acc --epochs=60

#### DenseGCN
python train_acc.py --exp_name=densegcn_acc --model_name=densegcn_acc --epochs=60

#### DilatedGCN
python train_acc.py --exp_name=dilated_acc --model_name=dilated_acc --epochs=60

#### MultiheadGCN
python train_acc.py --exp_name=multiheadgcn_acc --model_name=multiheadgcn --epochs=60

# Acknowledgement
https://github.com/chjchjchjchjchj/3D-point-classification/tree/master
DGCNN(https://github.com/WangYueFt/dgcnn/blob/master/pytorch/) by WangYueFt
