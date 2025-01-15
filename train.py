"""
任务: 完成基础的提取加载数据函数,实现数据迭代器以及train函数
时间: 2024/10/31-Redal
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from tensorboardX import SummaryWriter
from utils import GestureDataset,compute_total_params
from utils import save_data_to_h5py,load_data_from_h5py
from utils import plot_confusion_matrix
from models.Video_CNN_LSTM import GestureLSTM
from models.Video_PointNet_LSTM import PointNetClassifier
from models.Video_Transformer_LSTM import VideoClassifierViT
from sklearn.metrics import classification_report


def save_features_label(dataset_dir, save_dir, custom_dict):
    """提取视频的特征点和标签保存在./embed文件夹下
    features: 特征点数据集(N, T, L, 2)
    labels: 标签数据集(N,)
    dataset_dir: 视频数据集目录
    save_dir: 保存特征点和标签的目录
    """
    videodata_processor = GestureDataset(dataset_dir,
        label_to_index_dict=custom_dict, shuffle_label=True)
    features, label = videodata_processor.extract_keypoints()
    save_data_to_h5py(save_dir, features, label)


def load_features_label(save_dir):
    """从./embed文件夹下读取特征点和标签,转为Tensor数据
    features: 特征点数据集(N, T, L, 2)
    labels: 标签数据集(N,)
    save_dir: 保存特征点和标签的目录
    """
    features, label = load_data_from_h5py(save_dir)
    features = torch.from_numpy(features).float()
    label = torch.from_numpy(label).long()
    return features, label



class TrainDataLoader(Dataset):
    """特征训练数据加载器
    features: 手势特征点坐标,特征点数据集(N, T, L, 2)
    labels: 标签数据集(N,)"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    

class VideoDataset(Dataset):
    """视频数据集加载器
    video_dir: 视频数据集目录
    label_to_index_dict: 标签到索引的映射字典"""
    def __init__(self, video_dir, label_to_index_dict):
        self.video_dir = video_dir
        self.label_to_index_dict = label_to_index_dict
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.video_list = [os.path.join(video_dir, cls_name, file_name) 
                        for cls_name in os.listdir(video_dir)  for file_name 
                        in os.listdir(os.path.join(video_dir, cls_name))]
        self.labels_index = [self.label_to_index_dict[os.path.basename(cls_name.split('_')[0])] 
                             for cls_name in self.video_list]
        self.num_frames, self.total_frames = 15, 46
        self.frame_interval = self.total_frames // self.num_frames
    def __len__(self):
        return len(self.video_list)
    def __getitem__(self, idx):
        self.video_cap = cv2.VideoCapture(self.video_list[idx])
        self.frames = [] # 用于保存抽帧的图片
        for i in range(self.total_frames):
            ret, frame = self.video_cap.read()
            if not ret: break
            elif i % self.frame_interval == 0 and len(self.frames) < self.num_frames:
                self.frames.append(self.transform(frame))
        self.video_cap.release()
        self.frames = torch.tensor(np.array(self.frames)).permute(1,0,2,3)
        self.label = torch.tensor(self.labels_index[idx])
        return self.frames, self.label


def train_model(model, train_loader, optimizer, scheduler,
                 criterion, num_epochs=200, model_name=None):
    """训练模型"""
    model.train().to(device)
    print(f'====Model Total Params: {compute_total_params(model)}')
    print('====Start Training====')
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (input, label) in enumerate(train_loader):
            input = input.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output, _ = model(input)
            # 计算损失
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        # 记录平均损失
        avg_loss = running_loss / len(train_loader)
        writer.add_scalar('loss',avg_loss, epoch)
        if (epoch + 1) % 50 == 0:
            print('====[%d] loss: %.3f' % (epoch + 1, 
                    running_loss / len(train_loader)))
        import torch # 确保不会被覆盖
        torch.save(model.state_dict(), model_name)
    print('====Finished Training====')


def test(model, test_loader, labael_to_index_dict, title=None, hmp=None):
    """对模型进行测试, 绘制混淆矩阵"""
    model.eval().to(device)
    correct , S_time= 0 , time()
    y_pred , y_true= [],[]
    total = len(test_loader.dataset)
    with torch.no_grad():
        for i, (input, label) in enumerate(test_loader):
            input, label = input.to(device), label.to(device)
            label = label.cpu().numpy()
            _, output_softmax = model(input)
            
            predicted = np.argmax(output_softmax.cpu().numpy())
            y_pred.append(predicted)
            y_true.append(label[0])
            correct += (predicted == label).sum()
    accuracy = 100 * correct / total
    E_time = time(); all_time =E_time - S_time
    print(f'====Test Accuracy: {accuracy:.2f}%, correct number: {correct}/{total}, All Time: {all_time:.2f}s')    
    
    # 绘制混淆矩阵
    classification_report_dict = classification_report(y_true, y_pred)
    print(classification_report_dict)
    title = f'{title}_{accuracy:.2f}%'
    plot_confusion_matrix(y_true, y_pred, labael_to_index_dict,
                           normalize=False, title=title,hmp=hmp )
    
    

if __name__ == '__main__':
    # 预先处理数据集
    label_to_dict = {'up':0, 'down':1, 'left':2, 'right':3, 'attack':4,'retreat':5,'circle':6,
                     'vectory':7, 'okay':8, 'takeoff':9, 'landing':10 ,'negative':11}
    label_clses = len(label_to_dict)
    data_dir = r'./dataset/train/video' # 提取训练集特征点数据
    save_file = rf'./embed/gesture_features_label_{len(label_to_dict)}.h5'
    save_features_label(data_dir, save_file, label_to_dict)
    feature, label = load_features_label(save_file)
    print(f'====Feature Size: {feature.shape}\n====Label Size: {label.shape}')

    # 训练准备
    writer = SummaryWriter(log_dir='logs')
    train_loader = DataLoader(TrainDataLoader(feature, label), batch_size=32, shuffle=True) # 基于mediapipe特征点坐标数据集
    # video_train_dataloader = DataLoader(VideoDataset(data_dir, label_to_dict), batch_size=1, shuffle=True) # 基于视频数据集
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    print(f'====Current Compute Device: {device}')

    cnn_lstm_model = GestureLSTM(input_size=448, hidden_size=128, num_layers=2, num_classes=label_clses)
    cnn_lstm_optimizer = optim.Adam(cnn_lstm_model.parameters(), lr=0.001, weight_decay=0.001)
    cnn_lstm_scheduler = optim.lr_scheduler.StepLR(cnn_lstm_optimizer, step_size=100, gamma=0.1)
    pointnet_model = PointNetClassifier(num_classes=label_clses, feature_transform=None)
    pointnet_optimizer = optim.Adam(pointnet_model.parameters(), lr=0.001, weight_decay=0.001)
    pointnet_scheduler = optim.lr_scheduler.StepLR(pointnet_optimizer, step_size=50, gamma=0.1)
    video_vit_model = VideoClassifierViT(num_classes=label_clses, num_points=21, in_chans=2, embed_dim=768,
                               depth=1, num_heads=2, mlp_ratio=4, qkv_bias=True,
                               qk_scale=None, drop_rate=0.3, attn_drop_rate=0.0, num_frames=46, dropout_rate=0.3)
    video_vit_model_optimizer = optim.Adam(video_vit_model.parameters(), lr=0.001, weight_decay=0.001)
    video_vit_model_scheduler = optim.lr_scheduler.StepLR(video_vit_model_optimizer, step_size=50, gamma=0.1)
    
    # 训练模型
    train_model(cnn_lstm_model, train_loader, cnn_lstm_optimizer, cnn_lstm_scheduler, 
                criterion, num_epochs=400, model_name=f'./state_dict/cnn_lstm_dynamic_{label_clses}.pth')
    train_model(pointnet_model, train_loader, pointnet_optimizer, pointnet_scheduler, 
                criterion, num_epochs=400, model_name=f'./state_dict/pointnet_dynamic_{label_clses}.pth')
    train_model(video_vit_model, train_loader, video_vit_model_optimizer, video_vit_model_scheduler, 
                criterion, num_epochs=400, model_name=f'./state_dict/video_vit_dynamic_{label_clses}.pth')

    # 测试模型
    data_dir = r'./dataset/test/video'  # 提取测试集特征点数据
    save_file = rf'./embed/gesture_features_label_{len(label_to_dict)}_test.h5'
    save_features_label(data_dir, save_file, label_to_dict)
    feature, label = load_features_label(save_file)
    test_loader = DataLoader(TrainDataLoader(feature, label), batch_size=1, shuffle=True)

    cnn_lstm_model.load_state_dict(torch.load(f'./state_dict/cnn_lstm_dynamic_{label_clses}.pth'))
    test(cnn_lstm_model, test_loader, label_to_dict, title='CNN_LSTM_Model', hmp='Blues')
    pointnet_model.load_state_dict(torch.load(f'./state_dict/pointnet_dynamic_{label_clses}.pth'))
    test(pointnet_model, test_loader, label_to_dict,title= 'PointNet_LSTM_Model', hmp='Reds')
    video_vit_model.load_state_dict(torch.load(f'./state_dict/video_vit_dynamic_{label_clses}.pth'))
    test(video_vit_model, test_loader, label_to_dict,title='Transformer_LSTM_Model', hmp='Greens')