# YOLOv5 TensorRT Camera Inference

这是一个基于YOLOv5和TensorRT的目标检测程序，用于视频文件和摄像头实时检测。

## 使用说明

### 视频文件检测


目前比较懒，camera和当前yolov5_det,一个是usb相机，一个是视频，替换一个名字就不用改CMakeLists，后边频繁使用，我就调整一下了
video detect：
```bash
./yolov5_det -d model.engine video.mp4
``` 
camera detect：
```bash
./yolov5_det -d model.engine camera
``` 
参数说明
-d：指定模型引擎文件。
model.engine：模型引擎文件的路径。
video.mp4：需要检测的视频文件。
camera：使用摄像头进行检测。

构建和运行
生成.wts文件：
从PyTorch模型生成.wts文件，或者从模型库中下载.wts文件。
```bash
git clone -b v7.0 https://github.com/ultralytics/yolov5.git
git clone -b yolov5-v7.0 https://github.com/wang-xinyu/tensorrtx.git
cd yolov5/
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
cp [PATH-TO-TENSORRTX]/yolov5/gen_wts.py .
python gen_wts.py -w yolov5s.pt -o yolov5s.wts
``` 
# 生成yolov5s.wts文件
构建TensorRT引擎并运行：
构建TensorRT引擎并运行检测程序。
```bash
cd [PATH-TO-TENSORRTX]/yolov5/
mkdir build
cd build
cp [PATH-TO-ultralytics-yolov5]/yolov5s.wts .
cmake ..
make

./yolov5_det -s yolov5s.wts yolov5s.engine s
./yolov5_det -d yolov5s.engine ../images

# 例如，使用自定义模型
./yolov5_det -s yolov5_custom.wts yolov5.engine c 0.17 0.25
./yolov5_det -d yolov5.engine ../images
``` 

参考
该项目基于以下GitHub项目的代码：

TensorRTx：https://github.com/wang-xinyu/tensorrtx
ultralytics/yolov5：https://github.com/ultralytics/yolov5