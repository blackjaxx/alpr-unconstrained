# 非受限场景下的车牌识别 (ALPR Unconstrained)

## 简介

本仓库是 ECCV 2018 论文 [*"License Plate Detection and Recognition in Unconstrained Scenarios"*](http://sergiomsilva.com/pubs/alpr-unconstrained/) 的实现代码。

> **v2.0.0 现代化升级**：已将工具链全面升级为 Python 3.11+ / Keras 3 / TensorFlow 2.x / YOLOv8 / PaddleOCR，不再需要编译 Darknet C 代码。同时提供了图形化 Web 界面。

如果您的出版物使用了本代码产生的成果，请引用我们的论文：

```bibtex
@INPROCEEDINGS{silva2018a,
  author={S. M. Silva and C. R. Jung},
  booktitle={2018 European Conference on Computer Vision (ECCV)},
  title={License Plate Detection and Recognition in Unconstrained Scenarios},
  year={2018},
  pages={580-596},
  doi={10.1007/978-3-030-01258-8_36},
  month={Sep},
}
```

## 环境要求

安装依赖非常简单，只需一行命令：

```bash
pip install .
```

这将自动安装以下依赖：
- **Keras 3 + TensorFlow 2.15+** — WPOD-Net 车牌检测网络
- **Ultralytics YOLOv8** — 车辆检测（替代原版 Darknet YOLO）
- **PaddleOCR** — 车牌字符识别（替代原版 Darknet OCR）
- **OpenCV 4.8+** — 图像处理
- **Flask 3.0+** — Web UI 框架
- **NumPy** — 数值计算

> 不再需要编译 Darknet C 代码，不再受 CUDA 版本兼容性困扰。

## 下载模型

执行以下脚本下载 WPOD-Net 车牌检测模型的预训练权重：

```bash
bash get-networks.sh
```

YOLOv8 和 PaddleOCR 的模型权重会在首次运行时自动下载。

## 图形化 Web 界面

本项目提供了一个简洁的 Web 图形界面，支持图片上传和一键识别：

```bash
# 安装依赖后启动
python webapp.py
```

然后在浏览器中打开 [http://127.0.0.1:5000](http://127.0.0.1:5000)

**Web UI 功能：**
- 拖拽或点击上传图片
- 一键运行完整车牌识别流程
- 显示标注结果（车辆框 + 车牌四边形 + 号码）
- 显示各阶段处理耗时
- 自适应深色主题

## 命令行运行

使用 `run.sh` 脚本运行车牌识别流程。需要 3 个参数：

- **输入目录 (-i)**：包含至少一张 JPG 或 PNG 格式的图片
- **输出目录 (-o)**：识别过程中会生成临时文件，处理完毕后自动清理
- **CSV 文件 (-c)**：指定输出的 CSV 结果文件

```bash
bash get-networks.sh && bash run.sh -i samples/test -o /tmp/output -c /tmp/output/results.csv
```

### 三阶段流水线

1. **车辆检测** (YOLOv8) — 检测图片中的车辆（轿车、公交车、卡车）
2. **车牌检测** (WPOD-Net) — 对每辆车检测车牌位置，支持倾斜/透视车牌
3. **字符识别** (PaddleOCR) — 对每个车牌区域识别字符，输出车牌号码

## 训练车牌检测器

WPOD-Net 车牌检测网络支持从零训练或微调。

`samples/train-detector` 目录中有 3 个标注样本作演示用途。要完整复现论文实验，需将该目录替换为完整训练集。

从零训练命令：

```bash
mkdir models
python create-model.py eccv models/eccv-model-scratch
python train-detector.py --model models/eccv-model-scratch --name my-trained-model \
    --train-dir samples/train-detector --output-dir models/my-trained-model/ \
    -op Adam -lr .001 -its 300000 -bs 64
```

如需微调，在 `--model` 参数中指定已有模型路径即可。

## 关于 GPU 与 CPU

WPOD-Net 车牌检测网络在 Keras/TensorFlow 上运行，自动支持 GPU（CUDA）。

YOLOv8 和 PaddleOCR 同样自动检测并使用可用的 GPU。

只需正常安装 TensorFlow 的 GPU 版本，框架会自动利用 NVIDIA GPU 加速推理。

## 项目结构

```
alpr-unconstrained/
├── webapp.py                     # Web 图形界面 (Flask)
├── templates/
│   └── index.html                # Web UI 页面模板
├── vehicle-detection.py          # 阶段1: 车辆检测 (YOLOv8)
├── license-plate-detection.py    # 阶段2: 车牌检测 (WPOD-Net)
├── license-plate-ocr.py          # 阶段3: 字符识别 (PaddleOCR)
├── create-model.py               # WPOD-Net 模型定义
├── train-detector.py             # 训练脚本
├── run.sh                        # 流水线入口脚本
├── get-networks.sh               # 下载预训练模型
├── annotation-tool.py            # 标注工具（GUI）
├── gen-outputs.py                # 结果可视化输出
├── pyproject.toml                # 依赖管理
└── src/                          # 核心库
    ├── label.py                  # 标注数据类
    ├── loss.py                   # 自定义损失函数
    ├── keras_utils.py            # Keras 模型加载/推理
    ├── data_generator.py         # 多线程数据生成器
    ├── sampler.py                # 数据增强
    ├── projection_utils.py       # 透视变换
    ├── drawing_utils.py          # 绘图工具
    └── utils.py                  # 通用工具
```
