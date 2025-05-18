（1）在config文件中添加图像路径和手工形态测量特征路径；并给定数据集的类别数和类别名称
（2）选择多种不同的融合策略：
在train_lm_img.py中加载不同的模型
S1: models——>ResNet_MLP_double_flow
S2: models——>ResNet_MLP_double_flow_gate
S3: models——>ResNet_MLP_double_flow_cross_self

（3）执行main_lm_img.py训练模型
（4）执行test.py文件加载训练好的模型，并测试新的样本