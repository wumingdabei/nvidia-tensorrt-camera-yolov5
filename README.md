# YOLOv5 TensorRT Camera Inference

这是一个基于YOLOv5和TensorRT的目标检测程序，用于视频文件和摄像头实时检测。

## 使用说明

### 视频文件检测

使用以下命令运行模型检测视频文件：

```bash
./yolov5_det -d model.engine video.mp4
摄像头检测
使用以下命令运行模型进行摄像头检测：

bash
复制代码
./yolov5_det -d model.engine camera
参数说明
-d：指定模型引擎文件。
model.engine：模型引擎文件的路径。
video.mp4：需要检测的视频文件。
camera：使用摄像头进行检测。
构建和运行
生成.wts文件：
从PyTorch模型生成.wts文件，或者从模型库中下载.wts文件。

bash
复制代码
git clone -b v7.0 https://github.com/ultralytics/yolov5.git
git clone -b yolov5-v7.0 https://github.com/wang-xinyu/tensorrtx.git
cd yolov5/
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
cp [PATH-TO-TENSORRTX]/yolov5/gen_wts.py .
python gen_wts.py -w yolov5s.pt -o yolov5s.wts
# 生成yolov5s.wts文件
构建TensorRT引擎并运行：
构建TensorRT引擎并运行检测程序。

bash
复制代码
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
参考
该项目基于以下GitHub项目的代码：

TensorRTx
ultralytics/yolov5
markdown
复制代码

保存文件后，将修改提交到Git仓库并推送到远程仓库：

1. **将修改添加到暂存区**：
   ```bash
   git add README.md
提交修改：

bash
复制代码
git commit -m "Add custom usage instructions and references"
推送到远程仓库：

bash
复制代码
git push origin main
完整的命令流程如下：

bash
复制代码
# 编辑README文件，添加使用说明

# 保存并提交README文件
git add README.md
git commit -m "Add custom usage instructions and references"
git push origin main
这样，你的README文件将包含完整的使用说明，并且清楚地注明了参考的源代码。


1/2

还是需要你把使用，参考都写到readme里面，我直接复制一下
ChatGPT
当然，以下是一个完整的README示例，包含详细的使用说明和参考来源：

markdown
复制代码
# YOLOv5 TensorRT Camera Inference

这是一个基于YOLOv5和TensorRT的目标检测程序，用于视频文件和摄像头实时检测。

## 使用说明

### 视频文件检测

使用以下命令运行模型检测视频文件：

```bash
./yolov5_det -d model.engine video.mp4
摄像头检测
使用以下命令运行模型进行摄像头检测：

bash
复制代码
./yolov5_det -d model.engine camera
参数说明
-d：指定模型引擎文件。
model.engine：模型引擎文件的路径。
video.mp4：需要检测的视频文件。
camera：使用摄像头进行检测。
构建和运行
生成.wts文件：
从PyTorch模型生成.wts文件，或者从模型库中下载.wts文件。

bash
复制代码
git clone -b v7.0 https://github.com/ultralytics/yolov5.git
git clone -b yolov5-v7.0 https://github.com/wang-xinyu/tensorrtx.git
cd yolov5/
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
cp ../tensorrtx/yolov5/gen_wts.py .
python gen_wts.py -w yolov5s.pt -o yolov5s.wts
# 生成yolov5s.wts文件
构建TensorRT引擎并运行：
构建TensorRT引擎并运行检测程序。

bash
复制代码
cd ../tensorrtx/yolov5/
mkdir build
cd build
cp ../../yolov5/yolov5s.wts .
cmake ..
make

./yolov5_det -s yolov5s.wts yolov5s.engine s
./yolov5_det -d yolov5s.engine ../images

# 例如，使用自定义模型
./yolov5_det -s yolov5_custom.wts yolov5.engine c 0.17 0.25
./yolov5_det -d yolov5.engine ../images
参考
该项目基于以下GitHub项目的代码：

TensorRTx
ultralytics/yolov5