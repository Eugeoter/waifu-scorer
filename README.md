# Waifu Scorer

## 介绍

Waifu Scorer 项目提供美学评分模型的训练和推理脚本。

美学评分模型架构为 MLP，简单、快速而有效。

## 安装

```bash
git clone https://github.com/Eugeoter/waifu-scorer
cd waifu-scorer
pip install -r requirements.txt
```

## 推理

### UI

使用以下命令启动 UI：

```bash
python api.py
```

若以此种方式启动，则模型默认为 Eugeoter/waifu-scorer-v2/waifu-scorer-v2-1.pth。该模型用于对动漫插画（特别是人像）进行评分。

或者，您可以制定其他模型，通过 `--model_path` 参数指定模型路径，例如：

```bash
python api.py --model_path /path/to/your/model.pth
```

这里，`/path/to/your/model.pth` 是您的模型路径。

### 单图推理

请根据脚本 run_predict.py 的例子进行单图推理。

### 批量推理

请根据脚本 run_eval.py 的例子进行批量推理。推理结果将通过 Gradio UI 展示。

## 训练

请根据脚本 run_train.py 的例子进行训练。

### 准备

训练一个美学评分模型需要准备一个数据集。数据集文件结构示意如下：

    ```
    ├── dataset
    │   ├── class1
    │   │   ├── img1.jpg
    │   │   ├── img1.txt
    │   │   ├── img2.jpg
    │   │   ├── img2.txt
    │   │   └── ...
    │   ├── class2
    │   │   ├── img3.jpg
    │   │   ├── img3.txt
    │   │   ├── img4.jpg
    │   │   ├── img4.txt
    │   │   └── ...
    │   └── ...
    ```

指示数据集图像美学评分的方式有很多。项目默认使用包含质量标签的图像标注文件来作为美学评分，其中图像的标注文件（.txt）中包含图像的美学质量，例如：

img1.txt

```txt
1boy, high quality, solo, ...
```

其中，质量标签 high quality 表示图像的美学质量，将被脚本中的 quality_rating 函数对应为美学评分，其质量和评分的对应关系如下所示：

| 质量标签         | 评分 |
| ---------------- | ---- |
| horrible quality | 0    |
| worst quality    | 0    |
| low quality      | 1.5  |
| normal quality   | 5    |
| high quality     | 7    |
| best quality     | 8.5  |
| amazing quality  | 10   |

或者，你可以自定义自己的评分解释函数，在 run_train.py / custom_rating 中制定。评分函数接受一个 ImageInfo 类型的对象，包含了图像数据的基本信息。评分函数应返回一个浮点数，表示图像的美学评分，推荐为 0.0~10.0 分，分数越高质量越好。

数据集可存放于多个不同路径。脚本将会（递归地）加载所有路径下的数据集。具体行为可参阅项目 [WaifuSet](https://github.com/Eugeoter/waifuset)
