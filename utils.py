"""
任务:   1.完成手势识别位置((N*M, T, L*2))以及标签(N*M,)的提取
        2.将手势识别位置,标签保存到h5py文件中
时间: 2024/10/27-Redal
"""
import os
import cv2
import numpy as np
import h5py
import mediapipe as mp
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def plot_confusion_matrix(y_true, y_pred,label_to_index_dict, normalize=True, title=None, hmp=None):
    """绘制混淆矩阵, 并返回混淆矩阵"""
    # 将索引转为标签
    index_to_label_dict = {v: k for k, v in label_to_index_dict.items()}
    y_true = [index_to_label_dict[index] for index in y_true]
    y_pred = [index_to_label_dict[index] for index in y_pred]
    cm = confusion_matrix(y_true, y_pred)
    # 绘制混淆矩阵, 保证混淆矩阵的美观性
    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if title is None: title = 'Confusion matrix'
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.4)
    sns.heatmap(cm, annot=True, cmap=hmp)
    
    tick_marks = np.arange(len(label_to_index_dict)) + 0.5
    plt.xticks(tick_marks, list(index_to_label_dict.values()),fontsize=10, ha='center')
    plt.yticks(tick_marks, list(index_to_label_dict.values()),rotation=0, fontsize=10,ha='right')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'./output_image/{title}.jpg')
    plt.show()

    
def save_data_to_h5py(data_save_path, gesture_pos, gesture_label):
    """将手势特征点数据集保存到h5py文件中
    data_save_path: h5py文件保存路径
    gesture_pos: 手势特征点数据集->(N*M, T, L*2)
    gesture_label: 手势类别标签->(N*M,)
    """
    with h5py.File(data_save_path, 'w') as f:
        f.create_dataset('gesture_position', data=gesture_pos)
        f.create_dataset('gesture_label', data=gesture_label)
        f.close()


def load_data_from_h5py(data_load_path):
    """从h5py文件中加载手势特征点数据集
    data_load_path: h5py文件路径
    """
    with h5py.File(data_load_path, 'r') as f:
        gesture_pos = f['gesture_position'][:]
        gesture_label = f['gesture_label'][:]
        f.close()
    return gesture_pos, gesture_label


def compute_total_params(model):
    """计算模型参数数量
    model: 待计算参数数量的实例模型
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def check_all_frames(data_dir):
    """检查数据集是否包含所有帧数"""
    data_path = [os.path.join(data_dir, clsname, video_path) 
                 for clsname in os.listdir(data_dir) for video_path 
                 in os.listdir(os.path.join(data_dir, clsname))]
    for file in tqdm(data_path, desc="====Checking all frames"):
        video_cap = cv2.VideoCapture(file)
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames != 46:
            print(f'===={file} has {total_frames} frames')
        video_cap.release()


def index_to_label(gesture_index, label_to_index_dict):
        """将索引转换为手势类别标签
        gesture_index: 索引->(N*M,)
        """
        index_to_label_dict = {v: k for k, v in label_to_index_dict.items()}
        gesture_label = [index_to_label_dict[index] for index in gesture_index]
        return gesture_label

def test_voideo_to_features(video_path, totaL_frame=46):
    """
    将视频转为模型可以接受的特征形状
    video_path: 视频路径
    return: (1, T, L, 2)
    """
    video_cap = cv2.VideoCapture(video_path)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames != totaL_frame:
        print(f'===={video_path} has {total_frames} frames')
    mp_hands = mp.solutions.hands
    hands_model = mp_hands.Hands( max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    gesture_pos = [] # 手势特征点坐标(T, L*2)
    
    while video_cap.isOpened():
        flag, frame = video_cap.read()
        if flag:
            w, h, _ = frame.shape 
            results_hands = hands_model.process(frame)
            if results_hands.multi_hand_landmarks:
                for h_ls in results_hands.multi_hand_landmarks:
                    gesture_pos.append(np.array([[lm.x*w, lm.y*h] 
                    for lm in h_ls.landmark], dtype=np.int32))
            else: gesture_pos.append(np.zeros((21, 2), dtype=np.int32))
        else: break
    video_cap.release()
    return np.array(gesture_pos).reshape(1, -1, 21, 2)



class GestureDataset(Dataset):
    def __init__(self, data_dir,  label_to_index_dict=None, shuffle_label=False, process_show_label = False):
        """
        data_dir: 视频数据集目录,文件夹应包括多个手势类别的视频样本
        transform: 数据增强方法,对数据进行增强
        shuffle_label: 是否打乱数据集顺序
        self.gesture_pos: 手势特征点数据集->(N*M, T, L*2)
        self.gesture_label: 手势类别标签->(N*M,)
        """
        self.data_dir = data_dir
        self.label_to_index_dict = label_to_index_dict
        self.gesture_clsname_dirlist = [os.path.join(data_dir, clsname) 
                        for clsname in os.listdir(data_dir)]
        self.video_path_pathlist = [os.path.join(clsname, video_path) 
                        for clsname in self.gesture_clsname_dirlist 
                        for video_path in os.listdir(clsname)]
        if shuffle_label: np.random.shuffle(self.video_path_pathlist)
        self.process_show_label = process_show_label

        self.mp_hands = mp.solutions.hands
        self.hands_model = self.mp_hands.Hands( max_num_hands=1, 
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.gesture_pos = [] # 手势特征点坐标(T, L*2)
        self.gesture_position = [] # 手势特征点坐标(N*M, T, L*2)
        self.gesture_label = [] # 手势类别标签
    def __len__(self):
        return len(self.video_path_pathlist)    
    def extract_keypoints(self):
        """提取data_dir文件夹下的所有音频数据"""
        for idx in tqdm(range(self.__len__()), desc="====Extracting hand keypoints"):
            video_cap = cv2.VideoCapture(self.video_path_pathlist[idx])
            self.gesture_label.append(os.path.basename(self.video_path_pathlist[idx]).split('_')[0])
            self.gesture_pos = []

            while video_cap.isOpened():
                success, frame = video_cap.read()
                if success:
                    results_hands = self.hands_model.process(frame)
                    w, h, _ = frame.shape
                    if results_hands.multi_hand_landmarks:
                        # 若视频帧被检测到手势keypoints
                        for h_ls in results_hands.multi_hand_landmarks:
                            self.gesture_pos.append(np.array([[lm.x*w, lm.y*h] for lm in h_ls.landmark], dtype=np.int32))
                    else:
                        # 若视频帧未被检测到手势keypoints
                        self.gesture_pos.append(np.zeros((21, 2), dtype=np.int32))

                    if self.process_show_label:
                        # 显示处理过程的视频流
                        cv2.imshow('frame', frame)
                        cv2.waitKey(16)
                else:
                    break
            
            # 提取每一帧图像的->(L, P)
            # Feature Size: torch.Size([300, 46, 21, 2]), Label Size: torch.Size([300])
            self.gesture_position.append(np.array(self.gesture_pos))
            # 提取每一帧图像的坐标->(L,)
            # self.gesture_pos = np.array(self.gesture_pos).reshape(total_frames,-1)
            # self.gesture_position.append(self.gesture_pos)
            # 释放开销
            video_cap.release()
            if self.process_show_label: cv2.destroyAllWindows()
        self.gesture_label = self.label_to_index()
        return np.array(self.gesture_position), np.array(self.gesture_label)

    def label_to_index(self):
        """将手势类别标签转换为索引
        self.gesture_label: 手势类别标签->(N*M,)
        """
        gesture_index = [self.label_to_index_dict[label] for label in self.gesture_label]
        return gesture_index



if __name__ == '__main__':
    # 测试代码
    data_dir = r'.\dataset\test\video'
    check_all_frames(data_dir)
    