import os
from waifuset.classes import Dataset, ImageInfo
from waifuset.utils.file_utils import smart_path
from waifu_scorer.train import train
from waifu_scorer.train_utils import quality_rating


def custom_rating(img_info: ImageInfo):
    r"""
    Customize your rating function here. / 在此处自定义你的评分函数。
    """
    if img_info.category in ('horrible', 'impurity', 'worst'):
        return 0
    elif img_info.category in ('duplicate', 'watermark'):
        return 1.25
    elif img_info.category == 'low':
        return 2.5
    else:
        return quality_rating(img_info)


def custom_filter(img_info: ImageInfo):
    r"""
    Customize your filter function here. / 在此处自定义你的筛选函数。
    """
    if img_info.source.stem == 'download-2':
        if not img_info.caption or not img_info.caption.quality:
            return False
    elif img_info.source.stem in ('download', 'preparation'):
        if img_info.aesthetic_score is None:
            return False
        elif img_info.aesthetic_score < 6.0:
            return False

    if img_info.category.startswith(('azami', 'celluloid', 'dainty wilder', 'background')):
        return False
    elif img_info.category in ('alchemy stars', 'azur lane', 'arknights'):
        if not img_info.caption or not img_info.caption.quality:
            return False

    return True


if __name__ == '__main__':
    dataset = Dataset(
        source=[
            r"/path/to/your/dataset-1",  # [需要修改] 数据集1的路径
            r"/path/to/your/dataset-2",  # [可选] 数据集2的路径
            r"/path/to/your/metadata.json",  # [可选] 数据集3的路径
        ],
        verbose=True,
    ).make_subset(condition=custom_filter)

    train(
        # 基本配置
        dataset_source=dataset,
        save_path=os.path.join(smart_path("./models/", "%date%/%index%"), 'MLP.pth'),  # ![需要修改] 训练后模型的保存路径

        cache_path=r"D:\AI\datasets\laion\cache\database_2024-01-01_768_1.h5",  # 缓存数据的路径，建议填写，通过预先准备来大大提高训练速度
        cache_to_disk=True,  # [可选] 否将数据缓存到磁盘，建议设为 True，通过预先准备来大大提高训练速度
        # resume_path="/path/to/your/model.pth", # [可选] 恢复训练的模型路径

        rating_func_type=custom_rating,  # [可修改] 评分类型，根据数据标注的类型，可从 'direct', 'label', 'quality' 中选择，默认为 'quality'

        # 基本训练参数
        num_train_epochs=1000,  # [可修改] 训练轮数
        learning_rate=2e-6,  # [可修改] 学习率
        train_batch_size=256,  # [可修改] 训练集的 batch size
        val_batch_size=512,  # [可修改] 测试集的 batch size
        val_every_n_epochs=1,  # [可修改] 每隔多少轮验证一次
        val_percentage=0.05,  # [可修改] 从训练集中分出多少比例作为验证集
        save_best_model=True,  # [可修改] 是否在训练过程中自动保存当前最好的模型

        # 高级训练参数
        clip_batch_size=4,  # [可修改] 编码器的 batch size（目前似乎没有明显效果）
        mixed_precision=None,  # [可修改] 使用混合精度训练，可从 'fp16', 'bf16' 中选择，默认为不启用，即精度为 'float'
        persistent_workers=False,  # 是否使用持久化的 workers。牺牲内存换取训练速度
        max_data_loader_n_workers=0,  # 最大的 workers 数量，大于 0 的值似乎有bug
        mlp_model_type='mlp',  # [谨慎修改] MLP 模型的类型，可从 'default', 'large' 中选择，默认为 'default'。注意，模型类型不同，所得到模型的结构也不同，进行推理时需要手动切换模型类型
        clip_model_name="ViT-L/14",  # [谨慎修改] CLIP 模型的名称，可从 'RN50', 'RN50x4', 'RN101', 'RN50x16', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14' 中选择，默认为 'ViT-L/14'。该选择会直接影响 input_size，请注意切换！
        input_size=768,  # [谨慎修改] CLIP 模型的输入尺寸，应与 `clip_model_name` 的模型对应。
        batch_norm=True,  # [谨慎修改] 是否使用 batch norm
    )
