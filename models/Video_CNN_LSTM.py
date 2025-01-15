"""
任务: 完成手势识别LSTM动态模型的构建
时间: 2024/10/30-Redal
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx import export
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TimeDistributed(nn.Module):
    """
    对输入进行时间维度的扩展,以适应LSTM的输入要求
    module: 要扩展的模块
    """
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
    
    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        batch_size, time_steps = x.size(0), x.size(1)
        x = x.contiguous().view(batch_size * time_steps, *x.size()[2:])
        y = self.module(x)
        y = y.contiguous().view(batch_size, time_steps, *y.size()[1:])
        return y


class GestureLSTM(nn.Module):
    """输入数据形状要求,使用CNN+LSTM,进行动态手势识别但是通道为1 ->(B, T, C, L, P)
    input_size: 输入尺寸
    hidden_size: LSTM隐藏层大小
    num_layers: LSTM层数
    num_classes: 分类数量
    """
    def __init__(self, input_size=448, hidden_size=128, num_layers=2, num_classes=4, dropout_rate=0.3):
        """"使用LSTM模型进行动态识别手势"""
        super(GestureLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # 卷积特征扩增# (B, T, C, L, P) -> (B, T, P, L, C)
        self.conv_block_1 = nn.Sequential(
                    TimeDistributed(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1)),
                    TimeDistributed(nn.BatchNorm2d(16)),
                    TimeDistributed(nn.ReLU()),
                    
                    TimeDistributed(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1)),
                    TimeDistributed(nn.BatchNorm2d(32)),
                    TimeDistributed(nn.ReLU()),
                    TimeDistributed(nn.Dropout(dropout_rate)))
        
        # 卷积层提取特征+特征融合(B, T, P, L, C)
        self.conv_block_2_1 = nn.Sequential(
                    TimeDistributed(nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=2)),
                    TimeDistributed(nn.BatchNorm2d(16)),
                    TimeDistributed(nn.ReLU()),
                    TimeDistributed(nn.MaxPool2d(kernel_size=2, stride=2)),
                    TimeDistributed(nn.Dropout(dropout_rate)),

                    TimeDistributed(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2)),
                    TimeDistributed(nn.BatchNorm2d(32)),
                    TimeDistributed(nn.ReLU()),
                    TimeDistributed(nn.MaxPool2d(kernel_size=2, stride=2)),
                    TimeDistributed(nn.Dropout(dropout_rate)))
        
        self.conv_block_2_2 = nn.Sequential(
                    TimeDistributed(nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5, stride=1, padding=3)),
                    TimeDistributed(nn.BatchNorm2d(16)),
                    TimeDistributed(nn.ReLU()),
                    TimeDistributed(nn.MaxPool2d(kernel_size=2, stride=2)),
                    TimeDistributed(nn.Dropout(dropout_rate)),

                    TimeDistributed(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=3)),
                    TimeDistributed(nn.BatchNorm2d(32)),
                    TimeDistributed(nn.ReLU()),
                    TimeDistributed(nn.MaxPool2d(kernel_size=2, stride=2)),
                    TimeDistributed(nn.Dropout(dropout_rate)))
        
        # 卷积层融合特征(B, T, P=64, L, C) -> (B, T, P=16, L, C)
        self.con_block_3 = nn.Sequential(
            TimeDistributed(nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1)),
            TimeDistributed(nn.BatchNorm2d(16)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.AvgPool2d(kernel_size=3, stride=1)),
            TimeDistributed(nn.Dropout(dropout_rate)))
        
        # LSTMs层
        self.lstm_4_1 = nn.LSTM(input_size=input_size, hidden_size=2*hidden_size, bidirectional=False,
                            num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.lstm_4_2 = nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, bidirectional=False,
                            num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout_4 = TimeDistributed(nn.Dropout(dropout_rate))
        self.bn_4 = nn.BatchNorm1d(hidden_size)
        self.fc_4_1 = nn.Linear(in_features=hidden_size, out_features=2*hidden_size)
        self.fc_4_2 = nn.Linear(in_features=2*hidden_size, out_features=num_classes)
    
    def forward(self, x):
        # 输入数据形状要求,使用CNN+LSTM
        # 进行动态手势识别但是通道为1 ->(B, T, C, L, P)
        x = x.unsqueeze(2)
        # 卷积特征提取
        x_1 = self.conv_block_1(x)
        x_1 = x_1.permute(0, 1, 4, 3, 2)

        x_2_1 = self.conv_block_2_1(x_1)
        x_2_2 = self.conv_block_2_2(x_1)
        x_2 = torch.cat((x_2_1, x_2_2), dim=2)

        # (B, T, P, L, C)->(B, T, F)
        x_3 = self.con_block_3(x_2)
        x_3 = x_3.view(x_3.size(0), x_3.size(1), -1)

        lstm_embed, (h,c) = self.lstm_4_1(x_3)
        lstm_embed, (h,c) = self.lstm_4_2(lstm_embed)
        lstm_embed = self.dropout_4(lstm_embed)

        # batch_size, time_steps, hidden_size取最后时刻
        lstm_embed = lstm_embed[:, -1, :]
        lstm_embed = self.fc_4_1(self.bn_4(lstm_embed))
        logits_ = self.fc_4_2(lstm_embed)
        softmax_ = nn.Softmax(dim=1)(logits_)
        return logits_, softmax_
        
        
if __name__ == '__main__':
    # 代码测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GestureLSTM(input_size=448, hidden_size=128, num_layers=2, num_classes=12).to(device)
    x = torch.randn((4,46,21,2)).to(device)
    y = model(x)
    print(f'====Model: {model}, device: {device}====')
    print(f'====Input Shape: {x.shape}====')
    print(f'====Output Shape: {y[1].shape}====')

    # 导出onnx模型进行可视化
    input = torch.randn((32,46,21,2)).to(device)
    export(model, input, "./models/CNN_LSTM.onnx")
