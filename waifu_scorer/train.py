# os.environ['CUDA_VISIBLE_DEVICES'] = "0"       # in case you are using a multi GPU workstation, choose your GPU here

import os
import torch
import random
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from accelerate import Accelerator
from typing import Literal, Callable, Optional, Union
from waifuset.utils import log_utils
from waifuset.classes import Dataset, ImageInfo
from . import mlp, utils, train_utils

StrPath = Union[str, Path]


def train(
    dataset_source,
    save_path,
    resume_path: StrPath = None,
    data_preprocessor: Optional[Callable[[ImageInfo], float]] = None,
    rating_func_type: Union[Callable[[ImageInfo], float], Literal['direct', 'label', 'quality']] = 'quality',
    num_train_epochs=50,
    learning_rate=1e-3,
    train_batch_size=256,
    shuffle=True,
    flip_aug=True,
    val_batch_size=512,
    val_every_n_epochs=1,
    val_percentage=0.05,  # 5% of the training data will be used for validation
    save_best_model=True,
    clip_batch_size=1,
    cache_to_disk: bool = False,
    cache_path: StrPath = None,
    mixed_precision=None,
    max_data_loader_n_workers: int = 4,
    persistent_workers=False,
    mlp_model_type: Literal['default', 'large'] = 'default',
    clip_model_name: str = "ViT-L/14",
    input_size: int = 768,
    batch_norm: bool = True,
):
    r"""
    :param dataset_source: any dataset source, e.g. path to the dataset.
    :param save_path: path to save the trained model.
    :param resume_path: path to the model to resume from.
    :param cache_to_disk: whether to cache the training data to disk.
    :param cache_path: path to the cached training data. If not exists, will be created from `dataset_source`. If exists, will be loaded from disk.
    :param num_train_epochs: number of training epochs.
    :param learning_rate: learning rate.
    :param train_batch_size: training batch size.
    :param val_batch_size: validation batch size.
    :param val_every_n_epochs: validation frequency.
    :param val_percentage: percentage of the training data to be used for validation.
    :param encoder_batch_size: batch size for encoding images.
    :param mixed_precision: whether to use mixed precision training.
    :param max_data_loader_n_workers: maximum number of workers for data loaders.
    :param persistent_workers: whether to use persistent workers for data loaders.
    :param input_size: input size of the model.
    """
    log_utils.info(f"prepare for training")
    accelerator = Accelerator(mixed_precision=mixed_precision)
    weight_dtype = train_utils.prepare_dtype(mixed_precision)
    device = accelerator.device
    max_data_loader_n_workers = min(max_data_loader_n_workers, os.cpu_count()-1)
    if callable(rating_func_type):
        rating_func = rating_func_type
    else:
        rating_func = train_utils.get_rating_func(rating_func_type)

    model2, preprocess = utils.load_clip_models(name=clip_model_name, device=device)  # RN50x64

    dataset = Dataset(dataset_source, verbose=True, condition=lambda img_info: img_info.image_path.is_file())
    if data_preprocessor:
        for img_key, img_info in dataset.items():
            img_info = data_preprocessor(img_info)
    keys = list(dataset.keys())
    random.shuffle(keys)
    dataset = Dataset({k: dataset[k] for k in keys})

    num_pos = 0
    num_neg = 0
    num_mid = 0
    for img_key, img_info in dataset.items():
        rating = rating_func(img_info)
        if rating == 10:
            num_pos += 1
        elif rating == 0:
            num_neg += 1
        else:
            num_mid += 1
    log_utils.info(f"num_pos: {num_pos} | num_mid: {num_mid} | num_neg: {num_neg}")

    train_size = int(len(dataset) * (1 - val_percentage))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = Dataset(dataset.values()[:train_size]), Dataset(dataset.values()[train_size:])

    log_utils.info(f"train_size: {train_size} | val_size: {val_size}")

    train_dataset, train_loader = train_utils.prepare_dataloader(
        train_dataset,
        batch_size=train_batch_size,
        clip_batch_size=clip_batch_size,
        model2=model2,
        preprocess=preprocess,
        input_size=input_size,
        rating_func=rating_func,
        shuffle=shuffle,
        flip_aug=flip_aug,
        cache_to_disk=cache_to_disk,
        cache_path=cache_path,
        max_data_loader_n_workers=max_data_loader_n_workers,
        persistent_workers=persistent_workers,
        device=device,
    )

    val_dataset, val_loader = train_utils.prepare_dataloader(
        val_dataset,
        batch_size=val_batch_size,
        clip_batch_size=clip_batch_size,
        model2=model2,
        preprocess=preprocess,
        rating_func=rating_func,
        shuffle=shuffle,
        flip_aug=flip_aug,
        cache_to_disk=cache_to_disk,
        cache_path=cache_path,
        max_data_loader_n_workers=max_data_loader_n_workers,
        persistent_workers=persistent_workers,
        device=device,
    )

    rating_stat = {}
    for i in range(len(train_dataset)):
        # to list
        ratings: torch.Tensor = train_dataset[i]['ratings']
        ratings = ratings.squeeze().tolist()
        for rating in ratings:
            if rating not in rating_stat:
                rating_stat[rating] = 0
            rating_stat[rating] += 1

    log_utils.info("rating_stat:\n", '\n'.join(f'{k}: {v}' for k, v in rating_stat.items()))

    # prepare model

    model: mlp.MLP = utils.load_model(resume_path, model_type=mlp_model_type, input_size=input_size, batch_norm=batch_norm, device=device, dtype=weight_dtype)

    # import prodigyopt
    # print(f"use Prodigy optimizer | {optimizer_kwargs}")
    # optimizer_class = prodigyopt.Prodigy
    # optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # choose the loss you want to optimize for
    criterion = nn.MSELoss(reduction='mean')
    criterion2 = nn.L1Loss(reduction='mean')

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    log_utils.info(f"device: {accelerator.device}")

    # training loop
    best_loss = 999  # best validation loss
    total_train_steps = len(train_loader) * num_train_epochs
    progress_bar = tqdm(range(total_train_steps), position=0, leave=True)

    class LossRecorder:
        def __init__(self):
            self.loss_list = []
            self.loss_total: float = 0.0

        def add(self, *, epoch: int, step: int, loss: float) -> None:
            if epoch == 0:
                self.loss_list.append(loss)
            else:
                self.loss_total -= self.loss_list[step]
                self.loss_list[step] = loss
            self.loss_total += loss

        @property
        def moving_average(self) -> float:
            return self.loss_total / len(self.loss_list)

    loss_recorder = LossRecorder()
    model.requires_grad_(True)
    save_on_end = False

    try:
        for epoch in range(num_train_epochs):
            model.train()
            losses = []
            losses2 = []
            for step, input_data in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)
                im_emb_arr: torch.Tensor = input_data['im_emb_arrs'].to(accelerator.device).to(dtype=weight_dtype)  # shape: (batch_size, input_size)
                rating: torch.Tensor = input_data['ratings'].to(accelerator.device).to(dtype=weight_dtype)  # shape: (batch_size, 1)

                # randomize the rating
                # rating_std = 0.5
                # rating = rating + torch.randn_like(rating) * rating_std

                # log_utils.debug(f"x.dtype: {x.dtype} | y.dtype: {y.dtype} | model.dtype: {model.dtype}")

                with accelerator.autocast():
                    output = model(im_emb_arr)

                loss = criterion(output, rating)

                accelerator.backward(loss)

                losses.append(loss.detach().item())

                optimizer.step()

                # if step % 1000 == 0:
                #     print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, step, loss.item()))
                #     # print(y)

                progress_bar.update(1)

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                pbar_logs = {
                    'lr': f"{lr_scheduler.get_last_lr()[0]:.3e}",
                    'epoch': epoch,
                    'loss': avr_loss,
                }
                progress_bar.set_postfix(pbar_logs)

            progress_bar.write('epoch %d | avg loss %6.2f' % (epoch, avr_loss))

            # validation
            if accelerator.is_main_process and epoch > 0 and epoch % val_every_n_epochs == 0:
                model.eval()
                with torch.no_grad():
                    losses = []
                    losses2 = []
                    for step, input_data in enumerate(val_loader):
                        # optimizer.zero_grad(set_to_none=True)
                        im_emb_arr = input_data['im_emb_arrs'].to(accelerator.device).to(dtype=weight_dtype)
                        rating = input_data['ratings'].to(accelerator.device).to(dtype=weight_dtype)

                        with accelerator.autocast():
                            output = model(im_emb_arr)
                        loss = criterion(output, rating)
                        lossMAE = criterion2(output, rating)
                        # loss.backward()
                        losses.append(loss.detach().item())
                        losses2.append(lossMAE.detach().item())
                        # optimizer.step()

                        # if step % 1000 == 0:
                        #     print('\tValidation - Epoch %d | Batch %d | MSE Loss %6.2f' % (epoch, step, loss.item()))
                        #     print('\tValidation - Epoch %d | Batch %d | MAE Loss %6.2f' % (epoch, step, lossMAE.item()))

                        # print(y)
                    current_loss = sum(losses)/len(losses)
                    s = [f"validation - epoch {log_utils.stylize(epoch, log_utils.ANSI.YELLOW)}"]
                    s.append(f"avg MSE loss {log_utils.stylize(current_loss, log_utils.ANSI.GREEN, format_spec='.4f')}")
                    s.append(f"avg MAE loss {log_utils.stylize(sum(losses2)/len(losses2), log_utils.ANSI.YELLOW, format_spec='.4f')}")
                    progress_bar.write(' | '.join(s))
                    # progress_bar.write('validation - epoch %d | avg MSE loss %6.4f' % (epoch, sum(losses)/len(losses)))
                    # progress_bar.write('validation - epoch %d | avg MAE loss %6.4f' % (epoch, sum(losses2)/len(losses2)))

                    if save_best_model and current_loss < best_loss:
                        best_loss = current_loss
                        progress_bar.write(f"best MSE val loss ({log_utils.stylize(best_loss, log_utils.ANSI.BOLD, log_utils.ANSI.GREEN)}) so far. saving model...")
                        best_save_path = Path(save_path).parent / f"{Path(save_path).stem}_best-MSE{best_loss:.4f}{Path(save_path).suffix}"
                        train_utils.save_model(model, best_save_path, epoch=epoch)
                        progress_bar.write(f"model saved: `{save_path}`")

            lr_scheduler.step()
            accelerator.wait_for_everyone()
    except KeyboardInterrupt:
        log_utils.warn("KeyboardInterrupt")
        if input(f"save model to {save_path}? [y/n]") == 'y':
            save_on_end = True
    else:
        save_on_end = True

    progress_bar.close()
    model = accelerator.unwrap_model(model)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process and save_on_end:
        log_utils.info("saving model...")
        train_utils.save_model(model, save_path)
        log_utils.info(f"model saved: `{save_path}`")

    del accelerator

    log_utils.success(f"training done. best loss: {best_loss}")

    # inferece test with dummy samples from the val set, sanity check
    # log_utils.info("inference test with dummy samples from the val set, sanity check")
    # model.eval()
    # output = model(x[:5].to(device))
    # log_utils.info(output.size())
    # log_utils.info(output)
