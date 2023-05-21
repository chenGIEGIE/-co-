import sys
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel,QMessageBox
from PyQt5.QtCore import Qt
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("风机预测软件")

        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(800, 600)

        # 添加“导入预测数据集”按钮
        self.import_features_button = QPushButton("导入预测数据集", self)
        self.import_features_button.setGeometry(50, 50, 200, 30)
        self.import_features_button.clicked.connect(self.import_features)

        # 添加“导入目标值数据集”按钮
        self.import_targets_button = QPushButton("导入目标值数据集", self)
        self.import_targets_button.setGeometry(50, 100, 200, 30)
        self.import_targets_button.clicked.connect(self.import_targets)

        # 显示导入的文件名
        self.features_label = QLabel("", self)
        self.features_label.setGeometry(300, 50, 400, 30)
        self.targets_label = QLabel("", self)
        self.targets_label.setGeometry(300, 100, 400, 30)

        # 添加“预测”按钮
        self.predict_button = QPushButton("预测", self)
        self.predict_button.setGeometry(50, 150, 200, 30)
        self.predict_button.clicked.connect(self.predict)

    def import_features(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("CSV files (*.csv)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.features_label.setText(file_path)

    def import_targets(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("CSV files (*.csv)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.targets_label.setText(file_path)

    def predict(self):
        # 加载模型
        model = joblib.load("rf.pkl")

        # 加载特征值数据集
        features_file = self.features_label.text()
        features_df = pd.read_csv(features_file)
        #加载实际值数据集
        targets_file=self.targets_label.text()
        targets_df=pd.read_csv(targets_file)
        # 进行预测
        pre=model.predict(features_df[['windSpeed_value','fan_off','fint_volume-SQ','fint_speed-S2','congestion','workday','co_value_1hours']])
        predictions = np.exp(pre) - 1
        predictions[predictions >= 18] = predictions[predictions >= 18] * 1.5

        targets=targets_df['co_value']
        # 生成折线图
        plt.figure()
        time_col = features_df["time"]
        plt.plot(time_col, predictions,label='Predictions',linewidth=1,alpha=0.8)
        plt.plot(time_col,targets,label="Targets",linewidth=1,alpha=0.8)
        # 设置x轴标签的间隔
        plt.xticks(time_col.index[::1000], time_col[::1000])
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Predictions")

        plt.show()

        # 在预测结果中查找大于25的索引
        predictions_series = pd.Series(predictions, index=features_df.index)
        high_predictions = predictions_series[predictions_series > 25]
        if len(high_predictions) > 0:
            high_indices = high_predictions.index
            high_times = time_col[high_indices]
            # 显示弹窗
            msg_box = QMessageBox()
            msg_box.setText(f"建议开启风机的时间为：{', '.join(str(x) for x in high_times)}")
            msg_box.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
