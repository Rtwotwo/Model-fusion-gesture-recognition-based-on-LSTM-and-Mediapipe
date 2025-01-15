"""
任务: 根据设计好的数据采集, 手势识别, 风格迁移系统综合进行设计
      实现多界面多功能的切换, 并通过视频流实现实时数据采集, 手势识别, 风格迁移
时间: 2024/11/14-Redal
"""
import os
import random
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from APP_Data_Collector import HumanGestureDataset
from APP_Gesture_Recognizer import GestureRecognizerApp


class Gesture_Style_APP:
    """手势识别-风格迁移APP
    针对不同功能的APP, 实现了三个子界面,
    并通过Combobox控件实现功能的切换."""
    def __init__(self):
        self.main_root = tk.Tk()
        self.main_root.title("Gesture Style Recognition-Redal")
        self.main_root.geometry("600x400")
        
        # 创建相关的两个界面
        self.data_collector_app = HumanGestureDataset(root=tk.Toplevel(self.main_root))
        self.gesture_recognizer_app = GestureRecognizerApp(root=tk.Toplevel(self.main_root))
        
        # 隐藏子界面
        self.data_collector_app.root.withdraw()
        self.gesture_recognizer_app.root.withdraw()
        
        # 创建Combobox控件分别对应相关的功能
        self.function_select_combobox = ttk.Combobox(self.main_root, values=
                        ["数据采集", "手势识别"], state="readonly")
        self.function_select_combobox.place(x=420, y=50, width=80)
        self.function_select_combobox.set('功能选择')
        self.function_select_combobox.bind("<<ComboboxSelected>>", self.__select_function__)

        self.main_root_label_iron = tk.Label(self.main_root, width=400, height=400)
        self.main_root_label_iron.place(x=0, y=0)
        iron_dir_path = r'output_image'
        self.iron_paths = [os.path.join(iron_dir_path, file) for file in os.listdir(iron_dir_path)]
        self.load_image(random.choice(self.iron_paths), self.main_root_label_iron)
        
        self.main_root_exit_button = tk.Button(self.main_root, text='退出系统', command=self.main_root.quit)
        self.main_root_exit_button.place(x=420, y=20, width=80, height=30)
        
    def load_image(self, path, label):
        img_iron = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), (400, 400))
        PIL_image = Image.fromarray(img_iron)
        photo = ImageTk.PhotoImage(PIL_image)
        label.configure(image=photo)
        label.image = photo  

    def __select_function__(self, event):
        """选择相关的功能界面"""
        try: 
            function_name = self.function_select_combobox.get()
            if function_name == "数据采集":
                self.__show_window__(self.data_collector_app)
            elif function_name == "手势识别":
                self.__show_window__(self.gesture_recognizer_app)
        except: pass
        
    def __show_window__(self, window):
        """隐藏所有窗口, 显示选定的窗口"""
        self.data_collector_app.root.withdraw()
        self.gesture_recognizer_app.root.withdraw()
        
        # 释放视频捕获资源
        if hasattr(self.data_collector_app, 'cap') and self.data_collector_app.cap is not None:
            self.data_collector_app.cap.release()
            self.data_collector_app.cap = None
        if hasattr(self.gesture_recognizer_app, 'video_cap') and self.gesture_recognizer_app.video_cap is not None:
            self.gesture_recognizer_app.video_cap.release()
            self.gesture_recognizer_app.video_cap = None
        
        # 显示选定的窗口
        window.root.deiconify()
        # 重新初始化视频捕获
        if window is self.data_collector_app:
            self.data_collector_app.cap = cv2.VideoCapture(0)
            self.data_collector_app.start_video_capture()
        elif window is self.gesture_recognizer_app:
            self.gesture_recognizer_app.video_cap = cv2.VideoCapture(0)
            self.gesture_recognizer_app.start_video_capture()

    def on_close(self):
        """关闭所有窗口并释放资源"""
        if hasattr(self.data_collector_app, 'cap') and self.data_collector_app.cap is not None:
            self.data_collector_app.cap.release()
        if hasattr(self.gesture_recognizer_app, 'video_cap') and self.gesture_recognizer_app.video_cap is not None:
            self.gesture_recognizer_app.video_cap.release()
        self.main_root.quit()


if __name__ == "__main__":
    # 测试代码
    app = Gesture_Style_APP()
    app.main_root.protocol("WM_DELETE_WINDOW", app.on_close)
    app.main_root.mainloop()