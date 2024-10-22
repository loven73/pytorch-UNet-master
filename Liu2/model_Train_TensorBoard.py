#val_epoch  for batch_idx  150左右 四行

import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader

from skimage import io

from time import time

from .utils import chk_mkdir, Logger, MetricList
from .dataset2 import ImageToImage2D, Image2D


def f1_score(y_out, y_batch):
    y_out = torch.argmax(y_out, dim=1)
    TP = torch.sum((y_out == 1) & (y_batch == 1)).double()
    FP = torch.sum((y_out == 1) & (y_batch == 0)).double()
    FN = torch.sum((y_out == 0) & (y_batch == 1)).double()
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1.item()

def iou(y_out, y_batch):
    y_out = torch.argmax(y_out, dim=1)
    TP = torch.sum((y_out == 1) & (y_batch == 1)).double()
    FP = torch.sum((y_out == 1) & (y_batch == 0)).double()
    FN = torch.sum((y_out == 0) & (y_batch == 1)).double()
    IOU = TP / (TP + FP + FN + 1e-8)
    return IOU.item()


class Model:
    """
    Wrapper for the U-Net network. (Or basically any CNN for semantic segmentation.)

    Args:
        net: the neural network, which should be an instance of unet.unet.UNet2D
        loss: loss function to be used during training
        optimizer: optimizer to be used during training
        checkpoint_folder: path to the folder where you wish to save the results
        scheduler: learning rate scheduler (optional)
        device: torch.device object where you would like to do the training
            (optional, default is cpu)
        save_model: bool, indicates whether or not you wish to save the models
            during training (optional, default is False)
    """
    def __init__(self, net: nn.Module, loss, optimizer, checkpoint_folder: str,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 device: torch.device = torch.device('cpu')):
        """
        Wrapper for PyTorch models.

        Args:
            net: PyTorch model.
            loss: Loss function which you would like to use during training.
            optimizer: Optimizer for the training.
            checkpoint_folder: Folder for saving the results and predictions.
            scheduler: Learning rate scheduler for the optimizer. Optional.
            device: The device on which the model and tensor should be
                located. Optional. The default device is the cpu.

        Attributes:
            net: PyTorch model.
            loss: Loss function which you would like to use during training.
            optimizer: Optimizer for the training.
            checkpoint_folder: Folder for saving the results and predictions.
            scheduler: Learning rate scheduler for the optimizer. Optional.
            device: The device on which the model and tensor should be
                located. Optional.
        """
        self.net = net
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.checkpoint_folder = checkpoint_folder
        chk_mkdir(self.checkpoint_folder)

        # 在__init__方法中初始化了SummaryWriter，它将用于将日志写入TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_folder, 'logs'))

        # moving net and loss to the selected device
        self.device = device
        self.net.to(device=self.device)
        try:
            self.loss.to(device=self.device)
        except:
            pass

    def fit_epoch(self, dataset, epoch_idx, n_batch=1, shuffle=False):
        """
        Trains the model for one epoch on the provided dataset.

        Args:
             dataset: an instance of unet.dataset.ImageToImage2D
             n_batch: size of batch during training
             shuffle: bool, indicates whether or not to shuffle the dataset
                during training

        Returns:
              logs: dictionary object containing the training loss
        """

        self.net.train(True)

        epoch_running_loss = 0

        for batch_idx, (X_batch, y_batch, *rest) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=shuffle)):

            # 确保y_batch的标签值在0到类别数-1之间
            y_batch = y_batch.long()  # 确保标签是整数类型

            X_batch = Variable(X_batch.to(device=self.device))
            y_batch = Variable(y_batch.to(device=self.device))

            # training
            self.optimizer.zero_grad()
            y_out = self.net(X_batch)
            y_out = torch.softmax(y_out, dim=1)  # 应用softmax
            training_loss = self.loss(y_out, y_batch)
            training_loss.backward()
            self.optimizer.step()
            epoch_running_loss += training_loss.item()
            # Log the training loss to TensorBoard
            self.writer.add_scalar('Loss/train', training_loss.item(), batch_idx + len(dataset) * (epoch_idx - 1))

        self.net.train(False)

        del X_batch, y_batch

        logs = {'train_loss': epoch_running_loss / (batch_idx + 1)}

        return logs

    def val_epoch(self, dataset, epoch_idx, n_batch=1, metric_list=MetricList({})):
        """
        Validation of given dataset.

        Args:
             dataset: an instance of unet.dataset.ImageToImage2D
             n_batch: size of batch during training
             metric_list: unet.utils.MetricList object, which contains metrics
                to be recorded during validation

        Returns:
            logs: dictionary object containing the validation loss and
                the metrics given by the metric_list object
        """

        self.net.train(False)
        metric_list.reset()
        running_val_loss = 0.0
        epoch_f1_scores = []
        epoch_ious = []

        for batch_idx, (X_batch, y_batch_original, *rest) in enumerate(DataLoader(dataset, batch_size=n_batch)):

            # if X_batch.shape[1] == 4:  # 如果输入是4通道的，则只取前3个通道
            #     X_batch = X_batch[:, :3, :, :]



            print("model: pre_val_epoch . y_batch = ", y_batch_original)

            # 创建一个与 y_target_original 形状相同的张量，用于存放修改后的标签
            y_batch = y_batch_original.clone()
            # 将 255 替换为 1
            y_batch[y_batch_original == 255] = 1

            # 确保y_batch的标签值在0到类别数-1之间
            y_batch = y_batch.long()  # 确保标签是整数类型

            print("model: mid_val_epoch . y_batch = ", y_batch)

            X_batch = Variable(X_batch.to(device=self.device))
            y_batch = Variable(y_batch.to(device=self.device))

            y_out = self.net(X_batch)
            y_out = torch.softmax(y_out, dim=1)  # 应用softmax
            print("model: later_val_epoch . y_out = ", y_out)
            print("model: later_val_epoch . y_batch = ", y_batch)
            training_loss = self.loss(y_out, y_batch)
            val_loss = self.loss(y_out, y_batch)
            running_val_loss += val_loss.item()
            running_val_loss += training_loss.item()
            metric_list(y_out, y_batch)

            # 计算F1分数和IoU
            f1_score_val = f1_score(y_out, y_batch)
            iou_val = iou(y_out, y_batch)
            epoch_f1_scores.append(f1_score_val)
            epoch_ious.append(iou_val)

            # Log the validation loss to TensorBoard
            self.writer.add_scalar('Loss/val', training_loss.item(), batch_idx + len(dataset) * epoch_idx)



        del X_batch, y_batch

        # logs = {'val_loss': running_val_loss/(batch_idx + 1),
        #         **metric_list.get_results(normalize=batch_idx+1)}

        # 计算平均F1分数和IoU
        avg_f1_score = sum(epoch_f1_scores) / len(epoch_f1_scores)
        avg_iou = sum(epoch_ious) / len(epoch_ious)

        # 记录平均F1分数和IoU到TensorBoard
        self.writer.add_scalar('Metrics/val/f1_score', avg_f1_score, epoch_idx)
        self.writer.add_scalar('Metrics/val/iou', avg_iou, epoch_idx)

        logs = {
            'val_loss': running_val_loss / (batch_idx + 1),
            'avg_f1_score': avg_f1_score,
            'avg_iou': avg_iou
        }

        return logs

    def fit_dataset(self, dataset: ImageToImage2D, n_epochs: int, n_batch: int = 1, shuffle: bool = False,
                    val_dataset: ImageToImage2D = None, save_freq: int = 100, save_model: bool = True,
                    predict_dataset: Image2D = None, metric_list: MetricList = MetricList({}),
                    verbose: bool = False):



        """
        Training loop for the network.

        Args:
            dataset: an instance of unet.dataset.ImageToImage2D
            n_epochs: number of epochs
            shuffle: bool indicating whether or not suffle the dataset during training
            val_dataset: validation dataset, instance of unet.dataset.ImageToImage2D (optional)
            save_freq: frequency of saving the model and predictions from predict_dataset
            save_model: bool indicating whether or not you wish to save the model itself
                (useful for saving space)
            predict_dataset: images to be predicted and saved during epochs determined
                by save_freq, instance of unet.dataset.Image2D (optional)
            n_batch: size of batch during training
            metric_list: unet.utils.MetricList object, which contains metrics
                to be recorded during validation
            verbose: bool indicating whether or not print the logs to stdout

        Returns:
            logger: unet.utils.Logger object containing all logs recorded during
                training
        """

        logger = Logger(verbose=verbose)
        min_loss = np.inf
        train_start = time()

        for epoch_idx in range(1, n_epochs + 1):
            train_logs = self.fit_epoch(dataset, epoch_idx=epoch_idx, n_batch=n_batch, shuffle=shuffle)
            val_logs = {}
            if val_dataset is not None:
                val_logs = self.val_epoch(val_dataset, epoch_idx=epoch_idx, n_batch=n_batch, metric_list=metric_list)

            # Log the epoch-wise train and val loss to TensorBoard
            self.writer.add_scalars('Epoch_Loss',{'train': train_logs['train_loss'], 'val': val_logs.get('val_loss', 0)}, epoch_idx)

            val_logs = {}
            if val_dataset is not None:
                val_logs = self.val_epoch(val_dataset, epoch_idx=epoch_idx, n_batch=n_batch, metric_list=metric_list)
                loss = val_logs['val_loss']
            else:
                loss = train_logs['train_loss']

            # 保存模型
            if save_model:
                # 保存最佳模型
                if loss < min_loss:
                    min_loss = loss
                    torch.save(self.net.state_dict(), os.path.join(self.checkpoint_folder, 'best_model.pt'))

                # 按照指定频率保存模型
                if epoch_idx % save_freq == 0:
                    torch.save(self.net.state_dict(),
                               os.path.join(self.checkpoint_folder, f'model_epoch_{epoch_idx}.pt'))

            epoch_end = time()
            logs = {'epoch': epoch_idx,
                    'time': epoch_end - train_start,
                    'memory': torch.cuda.memory_allocated(),
                    **val_logs, **train_logs}
            logger.log(logs)
            logger.to_csv(os.path.join(self.checkpoint_folder, 'logs.csv'))

            # 保存预测结果
            if save_freq and (epoch_idx % save_freq == 0) and predict_dataset:
                epoch_save_path = os.path.join(self.checkpoint_folder, str(epoch_idx).zfill(4))
                chk_mkdir(epoch_save_path)
                torch.save(self.net.state_dict(), os.path.join(epoch_save_path, 'model.pt'))
                self.predict_dataset(predict_dataset, epoch_save_path)

        self.logger = logger

        # Close the TensorBoard writer
        self.writer.close()

        return logger

    def predict_dataset(self, dataset, export_path):
        """
        Predicts the images in the given dataset and saves it to disk.

        Args:
            dataset: the dataset of images to be exported, instance of unet.dataset.Image2D
            export_path: path to folder where results to be saved
        """
        self.net.train(False)
        chk_mkdir(export_path)

        for batch_idx, (X_batch, *rest) in enumerate(DataLoader(dataset, batch_size=1)):
            # if X_batch.shape[1] == 4:  # 如果输入是4通道的，则只取前3个通道
            #     X_batch = X_batch[:, :3, :, :]

            if isinstance(rest[0][0], str):
                image_filename = rest[0][0]
            else:
                image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

            # 修改mask的名称，确保它与原始图像的名称相同但扩展名为.png
            mask_filename = os.path.splitext(image_filename)[0] + '.png'

            X_batch = Variable(X_batch.to(device=self.device))
            y_out = self.net(X_batch).cpu().data.numpy()

            y_out_int = (y_out[0, 1, :, :] * 255).astype(np.uint8)

            # 然后使用转换后的数组保存图像
            io.imsave(os.path.join(export_path, image_filename), y_out_int)
