"""
任务: 手势识别软件GUI设计, 主题包含深度学习框架和几何图形两种策略
      深度学习框架: 依靠手势特侦点和vision transformer等模型实现手势识别
      几何图形: 定义检测区域和手势视觉中心(x, y, z)依靠手势特征点的位置变化实现手势识别
时间: 2024/11/13-Redal
"""
import os
import tkinter as tk
from tkinter import ttk
import cv2
import queue
import threading
import mediapipe as mp
from PIL import Image, ImageTk

import torch 
from models.Video_CNN_LSTM import GestureLSTM
from models.Video_PointNet_LSTM import PointNetClassifier
from models.Video_Transformer_LSTM import VideoClassifierViT
from utils import *



class GestureRecognizerApp(tk.Frame):
    """
    用于手势实时检测与分类软件
    frame: 用于显示视频的时序帧
    gesture_label: 用于显示当前手势的标签
    model_path: 模型路径
    """
    def __init__(self, root=None):
        super().__init__()
        # 初始视频流参数
        self.root = root
        self.video_cap = cv2.VideoCapture(0)
        self.thread = threading.Thread(target=self.video_loop)
        self.thread.daemon = True
        self.thread.start()
        self.set_main_widgets()
        self.gesture_position = np.array(np.zeros((46, 21, 2)), dtype=np.int32)

        # 加载检测模型
        self.label_to_index = {'up':0, 'down':1, 'left':2, 'right':3, 'attack':4,'retreat':5,'circle':6,
                     'vectory':7, 'okay':8, 'takeoff':9, 'landing':10 ,'negative':11}
        self.index_to_label = {v:k for k,v in self.label_to_index.items()}
        label_clses_num = len(self.label_to_index)
        self.cnn_lstm_model = GestureLSTM(num_classes=label_clses_num).eval()
        self.cnn_lstm_model.load_state_dict(torch.load(rf'./state_dict/cnn_lstm_dynamic_{label_clses_num}.pth'))
        self.pointnet_model = PointNetClassifier(num_classes=label_clses_num).eval()
        self.pointnet_model.load_state_dict(torch.load(rf'./state_dict/pointnet_dynamic_{label_clses_num}.pth'))
        self.transformer_model = VideoClassifierViT(num_classes=label_clses_num, num_points=21, in_chans=2,
                                embed_dim=768,depth=1, num_heads=2, mlp_ratio=4, qkv_bias=True,qk_scale=None, 
                                drop_rate=0.3, attn_drop_rate=0.0, num_frames=46, dropout_rate=0.3).eval()
        self.transformer_model.load_state_dict(torch.load(rf'./state_dict/video_vit_dynamic_{label_clses_num}.pth'))
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.toggle_dynamic_mode_active = False
        self.toggle_static_mode_active = False
        
    def video_loop(self):
        """视频流循环"""
        while True:
            flag, frame = self.video_cap.read()
            if flag:
                self.frame = cv2.flip( cv2.resize( cv2.cvtColor(frame,
                                cv2.COLOR_BGR2RGB), (400, 400)), 1)
                
                if self.toggle_dynamic_mode_active:
                    # 正常视频播放 
                    self.__label_to_show_video__()
                    self.__fourier_transform__()
                    if self.gesture_position.shape[0] >=46: 
                        # 特征点队列满46帧，删除第一个
                        self.gesture_position = self.gesture_position[1:]
                    # 手势特征点提取
                    w, h, _ = self.frame.shape
                    mp_hand_results = self.hands.process(self.frame)
                    if mp_hand_results.multi_hand_landmarks:
                        hand_landmarks = mp_hand_results.multi_hand_landmarks[0]
                        self.gesture_position =np.append(self.gesture_position, 
                                    np.array([[lm.x*w, lm.y*h] for lm in hand_landmarks.landmark],
                                    dtype = np.int32).reshape(1, 21, 2), axis=0)
                    else: self.gesture_position = np.append(self.gesture_position, 
                                    np.array(np.zeros((1, 21, 2)), dtype=np.int32),axis=0)
                    # 时序手势分类  
                    input_gesture_pos = torch.tensor(np.array(self.gesture_position
                                    ).reshape(1, -1, 21, 2), dtype=torch.float32)
                    _, probs_cnn_lstm = self.cnn_lstm_model(input_gesture_pos)
                    # _, probs_pointnet = self.pointnet_model(input_gesture_pos)
                    _, probs_transformer = self.transformer_model(input_gesture_pos)
                    gesture_index = np.argmax( probs_cnn_lstm.detach().numpy()*1/2 + 
                                        probs_transformer.detach().numpy()*1/2 )
                    # gesture_index = np.argmax(probs_transformer.detach().numpy())
                    gesture_name = self.index_to_label[gesture_index]
                    # 显示手势标签
                    if gesture_name != 'negative':
                        self.gesture_label.config(text=f'当前手势：{gesture_name}')
                    elif gesture_name == 'negative':
                        self.gesture_label.config(text='当前手势：无')
                        
                elif self.toggle_static_mode_active:
                    # TODO: 首先预测手势
                    if self.gesture_position.shape[0] >=46: 
                        # 特征点队列满46帧，删除第一个
                        self.gesture_position = self.gesture_position[1:]
                    # 手势特征点提取
                    w, h, _ = self.frame.shape
                    mp_hand_results = self.hands.process(self.frame)
                    if mp_hand_results.multi_hand_landmarks:
                        hand_landmarks = mp_hand_results.multi_hand_landmarks[0]
                        self.gesture_position =np.append(self.gesture_position, 
                                    np.array([[lm.x*w, lm.y*h] for lm in hand_landmarks.landmark],
                                    dtype = np.int32).reshape(1, 21, 2), axis=0)
                    else: self.gesture_position = np.append(self.gesture_position, 
                                    np.array(np.zeros((1, 21, 2)), dtype=np.int32),axis=0)
                    # 时序手势分类  
                    input_gesture_pos = torch.tensor(np.array(self.gesture_position
                                    ).reshape(1, -1, 21, 2), dtype=torch.float32)
                    _, probs_cnn_lstm = self.cnn_lstm_model(input_gesture_pos)
                    _, probs_pointnet = self.pointnet_model(input_gesture_pos)
                    _, probs_transformer = self.transformer_model(input_gesture_pos)
                    gesture_index = np.argmax( (probs_cnn_lstm.detach().numpy() + 
                            probs_transformer.detach().numpy()+probs_pointnet.detach().numpy())/3 )
                    # gesture_index = np.argmax(probs_cnn_lstm.detach().numpy())
                    gesture_name = self.index_to_label[gesture_index]
                    
                    
                    # 视频播放
                    hand_results = self.hands.process(self.frame)
                    h, w, _ = self.frame.shape
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            # 获取手势坐标(x, y, z)
                            temp_xy = [[lm.x*w, lm.y*h] for lm in hand_landmarks.landmark]
                            temp_z = [[lm.z] for lm in hand_landmarks.landmark]
                        hand_points_xy = np.array(temp_xy, dtype= np.int32)
                        hand_points_z = np.array(temp_z, dtype= np.float32)
                        del temp_xy, temp_z #释放内存
                    else:
                        hand_points_xy = np.zeros((21, 2), dtype=np.int32)
                        hand_points_z = np.zeros((21, 1), dtype=np.float32)
                    
                    # 特征点处理-获取方位角以及高度
                    mean_xy = np.mean(hand_points_xy, axis=0)
                    mean_z = np.mean(hand_points_z, axis=0)
                    first_direction = self.__frame_plot_shape__(mean_xy=mean_xy, mean_z=mean_z)
                    # 手势识别指令
                    if gesture_name == 'takeoff':
                        second_direction = '上升5m/s'
                    elif gesture_name == 'landing':
                        second_direction = '下降5m/s'
                    elif gesture_name == 'up':
                        if first_direction == '原地': second_direction = '休息'
                        else:                        second_direction = '平飞10m/s'
                    elif gesture_name == 'down':
                        if first_direction == '原地': second_direction = '休息'
                        else:                        second_direction = '平飞15m/s'
                    elif gesture_name == 'left':
                        if first_direction == '原地': second_direction = '休息'
                        else:                        second_direction = '平飞20m/s'
                    elif gesture_name == 'right':
                        if first_direction == '原地': second_direction = '休息'
                        else:                        second_direction = '平飞25m/s'
                    else: 
                        if first_direction == '原地': second_direction = '休息'
                        else:                        second_direction = '平飞5m/s'
                    self.gesture_label.config(text=f'当前手势：\n{first_direction}{second_direction}')

                    self.__label_to_show_video__()
                    self.__fourier_transform__()
                    
                else:
                    # 未选择模式
                    self.__label_to_show_video__()
                    self.gesture_label.config(text='请选择手势\n识别模式')
                    self.fourier_transform_label.config(text='等待选择手势\n识别模式...')
                    # 清除Fourier变换图片
                    self.fourier_transform_label.config(image='')
                    self.fourier_transform_label.image = None
            else: break
    
    def set_main_widgets(self):
        """设置窗口组件"""
        self.root.title("Gesture Recognizer App-Sub Redal")
        self.root.geometry("600x400")

        self.frame_label = tk.Label(self.root)
        self.frame_label.place(x=0, y=0, width=400, height=400)

        self.fourier_transform_label = tk.Label(self.root, font=('仿宋', 12))
        self.fourier_transform_label.place(x=400, y=200, width=200, height=200)

        self.gesture_label = tk.Label(self.root, text='请选择手势\n识别模式', font=('仿宋', 14))
        self.gesture_label.place(x=400, y=40, width=200, height=40)
        self.system_name_label = tk.Label(self.root, text='手势识别',font=('仿宋', 16))
        self.system_name_label.place(x=400, y=0, width=200, height=40)
        
        self.button_exit = tk.Button(self.root, text='退出', command=lambda: self.root.quit())
        self.button_exit.place(x=530, y=100, width=60, height=30)
        self.button_dynamic_mode = tk.Button(self.root, text='静态模式', command= self.__toggle_dynamic_mode__)
        self.button_dynamic_mode.place(x=410, y=100, width=60, height=30)
        self.button_position_mode = tk.Button(self.root, text='动态模式', command= self.__toggle_static_mode__)
        self.button_position_mode.place(x=470, y=100, width=60, height=30)


    def __label_to_show_video__(self):
        """在固定位置现实视频self.frame_label"""
        try:
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(self.frame))
            self.frame_label.config(image=img_tk)
            self.frame_label.image = img_tk
            self.root.after(33)
        except: pass

    def __on_sub_window_close__(self):
        """子窗口关闭时，恢复主窗口"""
        self.sub_data_root.destroy()
        self.root.deiconify()
    
    def __fourier_transform__(self):
        # 获取图像的尺寸
        image_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        rows, cols= image_gray.shape
        # 扩展图像到最优尺寸并计算Fourier变换
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        nimg = np.zeros((nrows, ncols))
        nimg[:rows, :cols] = image_gray
        dft = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        # 显示频谱图
        img_fourier_tk =ImageTk.PhotoImage(image=Image.fromarray(magnitude_spectrum))
        self.fourier_transform_label.config(image=img_fourier_tk)
        self.fourier_transform_label.image = img_fourier_tk

    def start_video_capture(self):
        if self.video_cap is None:
            self.video_cap = cv2.VideoCapture(0)
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.video_loop)
            self.thread.daemon = True
            self.thread.start()
    
    def __frame_plot_shape__(self, mean_xy=None, mean_z=None):
        """绘制检测手势区域, 用于后续的手势的静态识别"""
        self.frame = cv2.circle(self.frame, (200,200), 150, (0, 0, 255), 2)
        self.frame = cv2.circle(self.frame, (200,200), 30, (0, 255, 0), 2)
        self.frame = cv2.circle(self.frame, (200,320), 30, (0, 255, 0), 2)
        self.frame = cv2.circle(self.frame, (80,200), 30, (0, 255, 0), 2)
        self.frame = cv2.circle(self.frame, (320,200), 30, (0, 255, 0), 2)
        self.frame = cv2.line(self.frame, (200, 0), (200, 400), (0, 0, 255), 2)
        self.frame = cv2.line(self.frame, (0, 200), (400, 200), (0, 0, 255), 2)
        self.frame = cv2.putText(self.frame, 'forward', (160, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        self.frame = cv2.putText(self.frame, 'back', (175, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        self.frame = cv2.putText(self.frame, 'left', (55, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        self.frame = cv2.putText(self.frame, 'right', (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if mean_xy[0]!=0 and mean_xy[1]!=0:
            self.frame = cv2.circle(self.frame, (int(mean_xy[0]), int(mean_xy[1])), 15, (0, 255, 0), -1)
            self.frame = cv2.putText(self.frame, 'myself', (int(mean_xy[0]), int(mean_xy[1])),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
        distance = np.array([np.sqrt((x-mean_xy[0])**2+(y-mean_xy[1])**2) for x,y in [[200,200],[200,320],[80,200],[320,200]]])
        if distance[np.argmin(distance)] < 30: 
            name_dict = {0:'向前', 1:'向后', 2:'向左', 3:'向右'}
            return name_dict[np.argmin(distance)]
        else: return '原地'
        

    def __toggle_dynamic_mode__(self):
        """启动动态手势识别模式"""
        self.toggle_static_mode_active = False
        self.toggle_dynamic_mode_active = not self.toggle_dynamic_mode_active
    def __toggle_static_mode__(self):
        """启动静态手势识别模式"""
        self.toggle_dynamic_mode_active = False
        self.toggle_static_mode_active = not self.toggle_static_mode_active


if __name__ == '__main__':
    # # 测试代码
    root = tk.Tk()
    gesture_app = GestureRecognizerApp(root=root)
    gesture_app.root.mainloop()