"""
任务: 使用PointNet++进行手势识别
      PointNet++目的保持对数据的无序性的操作
时间: 2024/11/02-Redal
"""
import torch
import torch.nn as nn
from torch.onnx import export
import torch.nn.functional as F


class PointNetFeatureExtractor(nn.Module):
      """
      数据特征提取,使用PointNet对输入的批次数据进行处理
      input_dim: 输入数据的维度
      feature_transform: 是否使用特征变换
      """
      def __init__(self, input_dim=2, feature_transform=False):
            super(PointNetFeatureExtractor, self).__init__()
            self.feature_transform = feature_transform

            self.conv1 = nn.Conv1d(input_dim, 64, 1)
            self.conv2 = nn.Conv1d(64, 128, 1)
            self.conv3 = nn.Conv1d(128, 1024, 1)

            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)

            if self.feature_transform:
                  self.fstn = STNkd(k=64)
      def forward(self, x):
            # B：批量大小，T：帧数，N：点数，D：尺寸->(B * T, D, N)
            B, T, N, D = x.size() 
            x = x.view(B * T, N, D).transpose(2, 1) 

            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))

            if self.feature_transform:
                  x = x.transpose(2, 1)
                  trans_feat = self.fstn(x)
                  x = x.transpose(2, 1)
                  x = torch.bmm(trans_feat, x)

            x = self.bn3(self.conv3(x))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(B, T, 1024)
            # 将所有帧的特征向量拼接在一起
            x = x.view(B, -1)
            return x


class STNkd(nn.Module):
      """
      输入点云数据的空间变换的T-Transform
      """
      def __init__(self, k=64):
            super(STNkd, self).__init__()
            self.conv1 = nn.Conv1d(k, 64, 1)
            self.conv2 = nn.Conv1d(64, 128, 1)
            self.conv3 = nn.Conv1d(128, 1024, 1)
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, k*k)

            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)
            self.k = k
      def forward(self, x):
            B, D, N = x.size()
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)

            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
            x = self.fc3(x)

            iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(B, 1)
            x = x + iden
            x = x.view(-1, self.k, self.k)
            return x


class PointNetClassifier(nn.Module):
      """
      使用PointNet改进版进行手势识别与分类效果的测试
      num_classes: 手势类别数
      feature_transform: 是否使用特征变换
      """
      def __init__(self, num_classes=5, feature_transform=False):
            super(PointNetClassifier, self).__init__()
            # PointNet特征提取网路
            self.feature_transform = feature_transform
            self.feature_extractor = PointNetFeatureExtractor(feature_transform=feature_transform)
            self.dropout = nn.Dropout(p=0.3)

            # LSTM分类器处理时序数据
            self.lstm1 = nn.LSTM(input_size=1024, hidden_size=512, num_layers=2, batch_first=True)
            self.lstm2 = nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True)
            self.lstm_dropout = nn.Dropout(p=0.3)
            self.bn1 = nn.BatchNorm1d(256)
            self.bn2 = nn.BatchNorm1d(128)
            self.fc1 = nn.Linear(512, 256)  # 调整输入维度
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, num_classes)
      def forward(self, x):
            B, T, N, D = x.size()
            x = self.feature_extractor(x)
            x = x.reshape(B, T, -1)

            x_lstm1, (h,c) = self.lstm1(x)
            x_lstm2, (h,c) = self.lstm2(x_lstm1)
            x_dropout = self.lstm_dropout(x_lstm2)

            x =x_dropout[:, -1, :]  # 取最后一个时间步的特征
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
            out_logits = self.fc3(x)
            out_softmax = F.softmax(out_logits, dim=1)
            return out_logits, out_softmax


if __name__ == "__main__":
      # 模型代码测试
      num_classes = 12
      feature_transform = False
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      model = PointNetClassifier(num_classes=num_classes, feature_transform=feature_transform).to(device)

      features = torch.randn(200, 46, 21, 2).to(device)
      labels = torch.randint(0, num_classes, (200,))
      outputs, _ = model(features)
      
      print(f'====feature shape: {features.shape}')
      print(f'=====label shape: {labels.shape}')
      print(f'====output shape: {outputs.shape}')

      # 导出onnx模型进行可视化
      input = torch.randn((32,46,21,2)).to(device)
      export(model, input, "./models/PointNet_LSTM.onnx")