import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget
from test import draw_image,process_image,read_coordinates,operation,simulations_init,destroy
class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        # 初始化所有的fuzzy
        simulations_init()
        # 初始化摄像头
        self.camera = cv2.VideoCapture(0)
        self.is_paused = False

        # 初始化界面
        self.init_ui()

    def init_ui(self):
        # 创建界面组件
        self.video_label = QLabel(self)
        self.pause_button = QPushButton('Pause', self)
        self.exit_button = QPushButton('Quit', self)

        # 连接按钮的点击事件到相应的方法
        self.pause_button.clicked.connect(self.toggle_pause)
        self.exit_button.clicked.connect(self.exit_app)

        # 创建界面布局
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.exit_button)

        # 设置布局
        self.setLayout(layout)

        # 创建定时器，用于更新摄像头帧
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 更新帧率，单位为毫秒

        # 设置窗口标题和大小
        self.setWindowTitle('Window controller')

        self.setGeometry(100, 100, 500, 600)
        self.setFixedSize(500, 600)
        self.show()

    def toggle_pause(self):
        # 切换暂停状态
        self.is_paused = not self.is_paused

        # 根据暂停状态更新按钮文本
        if self.is_paused:
            self.pause_button.setText('Start')
        else:
            self.pause_button.setText('Pause')

    def exit_app(self):
        # 停止定时器、释放摄像头资源，并关闭应用程序
        self.timer.stop()
        self.camera.release()
        self.close()
        destroy()


    def update_frame(self):
        # 从摄像头读取帧
        ret, frame = self.camera.read()
        # 如果没有暂停且读取成功，则进行图像处理和显示
        if not self.is_paused and ret:
            processed_frame, results = process_image(frame)
            if results.multi_hand_landmarks:
                draw_image(processed_frame, results)
                read_coordinates(processed_frame,results)
            operation(results)
            # 将处理后的帧转换为Qt支持的格式
            height, width, channel = processed_frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(processed_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image.rgbSwapped())
            # 在界面上显示处理后的图像
            self.video_label.setPixmap(pixmap)
