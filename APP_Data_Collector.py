"""
任务: 完成固定时间(帧数)视频数据集的分类制作
时间: 2024/10/26-Redal
"""
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
from PIL import Image, ImageTk
from time import time
import mediapipe as mp


class HumanGestureDataset(tk.Frame):
    """
    数据采集软件,支持不同时间、不同类别采样视频和图片
    初始化参数:
        root: tkinter的主窗口
        mp_drawing: mediapipe的绘图工具
        thread: 线程对象操作工具
        stop_thread: 线程停止信号
    """
    def __init__(self, root=None, title="Data Collector App-Sub Redal", geometry='600x400'):
        super().__init__()
        self.root = root
        self.root.title(title)
        self.root.geometry(geometry)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands_model = self.mp_hands.Hands(max_num_hands=1, 
                        min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
        self.frame = None
        self.save_img = None
    
        self.cap = cv2.VideoCapture(0)
        self.video_time = 1.5  # 视频采集时间
        self.FPS = int(self.cap.get(cv2.CAP_PROP_FPS)) # 视频帧率FPS
        self.frame_count = self.video_time * self.FPS  # 需要保存的视频帧数
        self.current_frame = 0
        self.start_time = time()
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = None
        self.out_path = None
        self.imag_save_path= None
        self.gesture_name = None
        self.class_names = os.listdir('./dataset/test/video')
        
        self.thread = threading.Thread(target=self.update_frame)
        self.thread.daemon = True
        self.stop_thread = False
        self.thread.start()
        self.toggle_start_collect_active = False # 开始视频采集按钮是否可用
        self.toggle_image_collect_active = False # 开始图片采集按钮是否可用
        self.set_widgets()

    def set_widgets(self):
        self.label_video = tk.Label(self.root) 
        self.label_video.place(x=0, y=0, width=400, height=400) # 视频显示区域

        self.button_video_start = tk.Button(self.root, text="视频采集", command=self.toggle_start_collect)
        self.button_video_start.place(x=420, y=20) # 开始采集按钮

        self.button_video_start = tk.Button(self.root, text="图片采集", command=self.toggle_image_collect)
        self.button_video_start.place(x=480, y=20) # 开始采集按钮

        self.button_add_new_gesture = tk.Button(self.root, text="新增类名", command=self.toggle_add_new_gesture)
        self.button_add_new_gesture.place(x=540, y=20) # 开始采集按钮

        self.dropdown_video_time = ttk.Combobox(self.root, values=['1s', '2s', '3s', '4s'], state="readonly")
        self.dropdown_video_time.set('时间'); self.dropdown_video_time.place(x=420, y=50, width=59) # 视频采集时间下拉框
        self.dropdown_video_time.bind("<<ComboboxSelected>>", self.change_video_time)

        self.dropdown_gesture_name = ttk.Combobox(self.root, values=self.class_names, state="readonly")
        self.dropdown_gesture_name.set('类名'); self.dropdown_gesture_name.place(x=420, y=70, width=59) # 视频采集时间下拉框
        self.dropdown_gesture_name.bind("<<ComboboxSelected>>", self.change_gesture_name)

        self.lable_time_show = tk.Label(self.root, text=f"点击'开始采集'\n在规定{self.video_time}秒内完成", font=("宋体", 14))
        self.lable_time_show.place(x=400, y=100, width=200, height=100) # 显示采集时间

        self.label_show_histogram = tk.Label(self.root, text="直方图显示", font=("宋体", 16))
        self.label_show_histogram.place(x=400, y=200, width=200, height=200) # 实时显示图像直方图

    def toggle_start_collect(self):
        try:
            self.toggle_start_collect_active = not self.toggle_start_collect_active
            self.out = None # 重置视频输出对象
            self.start_time = time() # 重置开始时间
            self.lable_time_show.config(text=f"点击'开始采集'\n在规定{self.video_time}秒内完成") # 重置显示采集时间
            self.current_frame = 0 # 重置当前帧数
        except: print('====请选择手势名称gesture_name')

    
    def toggle_image_collect(self):
        """开始图片采集"""
        self.toggle_image_collect_active = not self.toggle_image_collect_active
        imag_save_dir = os.path.join("./dataset/train/image", self.gesture_name)
        if not os.path.exists(imag_save_dir):  os.makedirs(imag_save_dir)
        image_index = str(len(os.listdir(imag_save_dir)))
        # 保存采集的图片
        self.imag_save_path = os.path.join(imag_save_dir, self.gesture_name + "_" + image_index + ".jpg")
        Image.fromarray(self.save_img).save(self.imag_save_path)
        

    def update_frame(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                self.frame = cv2.flip( cv2.cvtColor(cv2.resize(frame, (400, 400)), cv2.COLOR_BGR2RGB), 1 )
                self.save_img = self.frame.copy()

                if self.toggle_start_collect_active:
                    """采集固定时间的视频作为数据集"""
                    if self.current_frame <= self.frame_count:
                        # 获取视频名称, 创建视频输出对象
                        if self.out is None:
                            if self.gesture_name is None: self.create_entry_box()
                            self.out = cv2.VideoWriter(self.out_path, self.fourcc, self.FPS, (400,400))
                        self.out.write(self.frame )
                        self.current_frame += 1
                        # 显示采集时间
                        elapsed_time = (self.frame_count - self.current_frame) / self.FPS
                        if elapsed_time > 0: text = f'采集视频\n剩余时间: {round(elapsed_time)}s'
                        else: text = f'采集视频\n已完成'
                        self.lable_time_show.config(text=text)
                        
                    else: self.toggle_star_collect_active = False
                    # 保证采集的视频能够被mediapepi手势检测到
                    results_hands = self.hands_model.process(self.frame)
                    if results_hands.multi_hand_landmarks:
                        for hand_landmarks in results_hands.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(self.frame, 
                                hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    # 显示采集的视频
                    self.common_video_play()

                else: 
                    results_hands = self.hands_model.process(self.frame)
                    if results_hands.multi_hand_landmarks:
                        for hand_landmarks in results_hands.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(self.frame, 
                                hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    self.common_video_play()
    
    def common_video_play(self):
        """进行常规视频的播放"""
        try:
            img = ImageTk.PhotoImage(image=Image.fromarray(self.frame))
            self.label_video.config(image=img)
            self.label_video.image = img
            self.show_image_hist() # 显示图像直方图
            self.after(33)
        except: pass

    def create_entry_box(self):
        """创建输入框, 以获取视频手势名称"""
        temp_root = tk.Toplevel(self.root)
        temp_root.geometry("200x100")
        temp_root.title("Redal")
        temp_entry = tk.Entry(temp_root)
        temp_entry.pack(pady=10)
        def get_entry_box_value():
            """获取输入, 构建视频存储路径"""
            self.gesture_name = str(temp_entry.get())
            out_pathdir = os.path.join("./dataset/test/video", self.gesture_name)
            if not os.path.exists(out_pathdir): os.makedirs(out_pathdir)
            video_index = str(len(os.listdir(out_pathdir)))
            self.out_path = os.path.join(out_pathdir, 
                self.gesture_name + "_" + video_index + ".avi")
            print(f'====视频输出路径: {self.out_path}')
            temp_root.destroy()
        temp_button = tk.Button(temp_root, text="确定", 
                    command=get_entry_box_value).pack(pady=10)
        self.root.wait_window(temp_root)
    
    def show_image_hist(self):
        """绘制图像的直方图并且进行显示"""
        channels = [0, 1, 2]
        colors = ['b', 'g', 'r']
        histograms = []
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        for channel, color in zip(channels, colors):
            hist = cv2.calcHist([self.frame], [channel], None, [256], [0, 256])
            histograms.append(hist)
        hist_normalized = [cv2.normalize(hist, hist, alpha=0, beta=255, 
            norm_type=cv2.NORM_MINMAX).astype(np.uint8) for hist in histograms]
        histr = np.ones((200, 256, 3), np.uint8)*255
        # 绘制直方图
        for i in range(1, 256):
            for channel, color in enumerate(colors):
                cv2.line(histr, (i - 1, 200 - int(hist_normalized[channel][i - 1])),
                          (i, 200 - int(hist_normalized[channel][i])), 
                          (int(color == 'b') * 255, int(color == 'g') * 255,
                             int(color == 'r') * 255), 1)
        histr = np.vstack((histr, np.ones((20, 256, 3), np.uint8)*255))
        img = cv2.resize(histr, (200,200), interpolation=cv2.INTER_CUBIC)
        img = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.label_show_histogram.config(image=img)
        self.label_show_histogram.image = img
    
    def change_video_time(self, event):
        """获取下拉框的视频时间选择"""
        self.video_time = int(self.dropdown_video_time.get()[0])
    
    def change_gesture_name(self, event):
        """获取下拉框的手势名称选择"""
        self.gesture_name = self.dropdown_gesture_name.get()
        out_pathdir = os.path.join("./dataset/test/video", self.gesture_name)
        if not os.path.exists(out_pathdir): os.makedirs(out_pathdir)
        video_index = str(len(os.listdir(out_pathdir)))
        self.out_path = os.path.join(out_pathdir, 
                self.gesture_name + "_" + video_index + ".avi")
        print(f'====视频输出路径: {self.out_path}')
    
    def toggle_add_new_gesture(self):
        """添加新的手势"""
        temp_root = tk.Toplevel(self.root)
        temp_root.geometry("200x100")
        temp_root.title("Redal")
        temp_entry = tk.Entry(temp_root)
        temp_entry.pack(pady=10)
        def get_entry_box_value():
            """获取输入, 构建视频存储路径"""
            self.gesture_name = str(temp_entry.get())
            self.class_names.append(self.gesture_name)
            self.set_widgets()
            temp_root.destroy()
        temp_button = tk.Button(temp_root, text="确定", 
                command=get_entry_box_value).pack(pady=10)
        self.root.wait_window(temp_root)

    def start_video_capture(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.update_frame)
            self.thread.daemon = True
            self.thread.start()


if __name__ == '__main__':
    root = tk.Tk()
    app = HumanGestureDataset(root=root)
    app.mainloop()