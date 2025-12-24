import os
import argparse
import sys
import time

import math
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from kan import KAN
from tqdm import tqdm

from logger import Logger
from losses import NMSE
from model.ARVTDNN import ARVTDNN
from model.LSTMModel import LSTMModel
from model.LSTM import LSTM
from model.RVTDCNN import RVTDCNN
from model.RVTDNN import RVTDNN
from myDataset import MyDataset

from utils.early_stopping import EarlyStopping
from utils.utils import *
from model.resnet import TimeSeriesResNet
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# file path
# /workspace/dataset/Nonlinear/D_frequency_band/Baseband_1G_2G_4G_1000VPP_16QAM
# /workspace/dataset/Nonlinear/D_frequency_band/Baseband_1G_2G_4G_1000VPP_64QAM
# /workspace/dataset/Nonlinear/satellite
parser = argparse.ArgumentParser()

parser.add_argument('--file_path', type=str,
                    default="dataset/")
parser.add_argument('--dataset', type=str, default="baseband_4G_20G_64QAM_5.mat")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--memory_depth', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--scheduler_step', type=int, default=20, help='Step size for scheduler')
parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='Gamma factor for scheduler')
parser.add_argument('--epochs', type=int, default=180)
parser.add_argument('--window_size', type=int, default=30)
parser.add_argument('--model_name', type=str, default="ResNet", choices=["LSTMModel","LSTM", "ResNet","RVTDNN","RVTDCNN","ARVTDNN","RVTDCNN"])
parser.add_argument('--use_features', action='store_true', default=False)
parser.add_argument('--shuffle', action='store_true', default=True)
parser.add_argument('--use_standard', action='store_true', default=False)
parser.add_argument('--use_label_standard', action='store_true', default=False)


args = parser.parse_args()

current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
log_name = f"./logs/{args.model_name}_{current_time}.txt"
sys.stdout = Logger(log_name)

if __name__ == '__main__':
    file_path = args.file_path
    data_name = args.dataset
    batch_size = args.batch_size
    window_size = args.window_size
    model_name = args.model_name
    device = torch.device("cuda:" + str(args.cuda) if torch.cuda.is_available() else "cpu")
    dataset = sio.loadmat(file_path + data_name)
    memorydepth = args.memory_depth + 1

    print(device)
    X = dataset['X'].flatten()  # 将X转换为一维数组


    len_X = len(X)  # X的长度
    all_data = []

    for i in range(len_X):
        # 获取当前项的前n项
        start_index = max(0, i - memorydepth + 1)
        end_index = i + 1
        segment = X[start_index:end_index]

        # 如果前面没有足够的项，用0占位
        if len(segment) < memorydepth:
            segment = np.concatenate((np.zeros(memorydepth - len(segment)), segment))
        if model_name in ["LSTMModel", "LSTM", "ARVTDNN", "RVTDCNN"]:
            # 反转segment的顺序
            segment = segment[::-1]  # 使用切片反转
        all_data.append(segment)

    # 将new_X转换为NumPy数组
    all_data = np.array(all_data)
    print(f"all data: {all_data.shape}")
    all_label = dataset['Y']
    print(f"all data: {all_data.shape}, all label: {all_label.shape}")
    for k, v in args.__dict__.items():
        print(f'{k}: {v}')
    if model_name == "LSTM" or model_name == "LSTMModel":
        all_data = np.stack((all_data.real, all_data.imag), axis=1)
        # 使用 transpose 重新排列维度
        all_data = all_data.transpose(0, 2, 1)

    elif model_name in ["ResNet"]:
        all_data = np.stack((all_data.real, all_data.imag), axis=1)
    elif model_name == "RVTDNN":
        all_data = np.concatenate((all_data.real, all_data.imag), axis=1)
    elif model_name == "ARVTDNN":
        all_data = np.concatenate((all_data.real, all_data.imag, abs(all_data),
                                   np.power(abs(all_data), 2), np.power(abs(all_data), 3)), axis=-1)
    elif model_name == "RVTDCNN":
        all_data = np.stack((all_data.real, all_data.imag, abs(all_data),
                                   np.power(abs(all_data), 2), np.power(abs(all_data), 3)), axis=1)
        all_data = all_data.transpose(0, 2, 1)
        # 使用 transpose 重新排列维度
    #
    all_label = np.concatenate((all_label.real, all_label.imag), axis=-1)
    print(f"all data: {all_data.shape}, all label: {all_label.shape}")

    train_data, test_data, train_label, test_label = train_test_split(all_data, all_label,
                                                                      test_size=0.2, shuffle=args.shuffle)
#    print(f"train data: {train_data.shape}, test data: {test_data.shape}")
   # print(f"train label: {train_label.shape}, test label: {test_label.shape}")

    train_dataset = MyDataset(torch.Tensor(train_data).float(), torch.Tensor(train_label).float())
    test_dataset = MyDataset(torch.Tensor(test_data).float(), torch.Tensor(test_label).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=args.shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=args.shuffle)
    if model_name == "LSTMModel":
        model = LSTMModel().to(device)
    elif model_name == "LSTM":
        model = LSTM().to(device)
    elif model_name == "ResNet":
        model = TimeSeriesResNet(train_data.shape[-2], train_label.shape[-1]).to(device)
    elif model_name == "RVTDNN":
        model = RVTDNN(args).to(device)
    elif model_name == "ARVTDNN":
        model = ARVTDNN(args).to(device)
    elif model_name == "RVTDCNN":
        model = RVTDCNN(args).to(device)
    criterion_doa = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    early_stopping = EarlyStopping(patience=20, verbose=True)
    train_loss = 0.0
    train_val = 0.0
    test_loss = 0.0
    eval_val = 0.0
    for epoch in tqdm(range(args.epochs), desc='Epochs'):
        model.train()
        total_loss = 0.0
        for batch_data, batch_label in train_loader:
            batch_data, batch_label = batch_data.to(device), batch_label.to(device)
            #print(f'Input shape to LSTM: {batch_data.shape}')
            output = model(batch_data)
            loss = criterion_doa(output, batch_label)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_loss = total_loss / len(train_loader)
        print(f'\nEpoch [{epoch + 1}/{args.epochs}], Train loss: {train_loss:.6f}')

        model.eval()
        train_val = 0.0
        eval_val = 0.0
        with torch.no_grad():
            val_loss = 0.0
            outputs = np.empty((0, test_label.shape[-1]))
            labels = np.empty((0, test_label.shape[-1]))
            for batch_data, batch_label in test_loader:
                batch_data, batch_label = batch_data.to(device), batch_label.to(device)
                output = model(batch_data)
                loss = criterion_doa(output, batch_label)
                val_loss += loss.item()
                outputs = np.append(outputs, output.detach().cpu().numpy(), axis=0)
                labels = np.append(labels, batch_label.detach().cpu().numpy(), axis=0)
            test_loss = val_loss / len(test_loader)
            eval_val = NMSE(outputs, labels)
            print(
                f'Validation loss: {val_loss / len(test_loader):.6f}, Validation NMSE: {math.log10(eval_val) * 10 :.6f}')

            outputs = np.empty((0, test_label.shape[-1]))
            labels = np.empty((0, test_label.shape[-1]))
            for batch_data, batch_label in train_loader:
                batch_data, batch_label = batch_data.to(device), batch_label.to(device)
                output = model(batch_data)
                outputs = np.append(outputs, output.detach().cpu().numpy(), axis=0)
                labels = np.append(labels, batch_label.detach().cpu().numpy(), axis=0)
            train_val = NMSE(outputs, labels)
            print(f'Train Dataset NMSE: {math.log10(train_val) * 10:.6f}')
        early_stopping(test_loss, train_loss, eval_val, train_val, model)
        if early_stopping.early_stop:
            test_loss = early_stopping.val_loss_min
            train_loss = early_stopping.train_loss
            eval_val = early_stopping.eval_val
            train_val = early_stopping.train_val
            model = early_stopping.model
            break

    save_time = time.strftime('%Y%m%d%H%M%S', time.localtime())

    sat_datasets = list(["PA_baseband_100M_QV.mat","PA_baseband_100M_Ka.mat","PA_baseband_200M_QV.mat","PA_baseband_200M_Ka.mat",
                         "baseband_1G_5G_16QAM.mat","baseband_2G_10G_16QAM.mat","baseband_4G_20G_16QAM.mat","baseband_1G_5G_64QAM.mat",
                         "baseband_2G_10G_64QAM.mat","baseband_4G_20G_64QAM.mat"])

    add_row = [f"{model_name}", f"{train_loss:.6f}", f"{train_val:.6f}", f"{math.log10(train_val) * 10:.6f}",
               f"{test_loss:.6f}", f"{eval_val:.6f}", f"{math.log10(eval_val) * 10:.6f}",
               f"{file_path + data_name}", f"{sat_datasets.index(data_name)}",f"{memorydepth}",
               f"{save_time}{args.shuffle}{args.use_features}{args.use_standard}{args.use_label_standard}"]
    write_csv("csv/logData.csv", add_row)
    model_path = f'saveModels/{model_name}_{save_time}.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Saved model to {model_path}')
