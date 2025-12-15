# 使用教程
## 1. 环境
cuda 11.8
tensorrt 8.6.1.6
opencv 4.13
## 2. 使用
1. 用gen_wts.py将.pt文件转化为.wts文件
2. 修改include/config.h文件
3. 修改build_xxx.cpp文件
4. 编译
5. 生成engine文件： `./yolo11_xxx`
6. 记得添加labels.txt文件
7. 推理： `./yolo11_infer <engine> <input(image|video|dir)> <labels.txt> [c|g postprocess] [save_result(0|1)]`
## 3.备注
一定要按照步骤来，该改的地方都要记得改
