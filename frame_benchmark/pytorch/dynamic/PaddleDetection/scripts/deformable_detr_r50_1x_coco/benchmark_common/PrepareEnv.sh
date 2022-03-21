#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`
pip install Cython
pip install torch==1.10.0 torchvision==0.11.1
# deformable-detr模型中包含自定义算子，只能运行在mmcv-full 1.4.2版本上
# 参考: https://github.com/open-mmlab/mmdetection/issues/7017
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
pip install -r requirements.txt
pip install -v -e .

################################# 准备训练数据 如:
wget -nc -P data/coco/ https://paddledet.bj.bcebos.com/data/coco_benchmark.tar
cd ./data/coco/ && tar -xf coco_benchmark.tar && mv -u coco_benchmark/* .
rm -rf coco_benchmark/ && cd ../../
echo "*******prepare benchmark end***********"