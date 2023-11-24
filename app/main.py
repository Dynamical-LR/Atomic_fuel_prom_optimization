from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QMainWindow, QLabel
from PyQt6 import QtCore

import sys
from baseline.model import transform, read, ForgererGroup, OvenSchedule

from PyQt6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsTextItem
from PyQt6.QtGui import QPainter, QColor, QFont

data = read('./train/day-56.json')
schedule = transform(data)

class GanttItem(QGraphicsRectItem):
    def __init__(self, name, start, duration, row_height, row_number):
        super().__init__(start, row_height * row_number, duration, 50)
        color = QColor(0, 120, 210)
        if name == 'kovka':
            color = QColor(255, 0, 0)
        elif name == 'podogrev':
            color = QColor(0, 255, 0)
        self.setBrush(color)  # Set the color of the Gantt item

        # Add a label for the task name
        self.label = QGraphicsTextItem(name, self)
        self.label.setPos(start, row_height * row_number + 10)
        font = QFont()
        font.setBold(True)
        self.label.setFont(font)

class GanttDiagram(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 1440, 2000)  # Adjusted width for 1440 minutes
        self.setWindowTitle('Gantt Diagram')

        scene = QGraphicsScene(self)
        view = QGraphicsView(scene, self)
        self.setCentralWidget(view)

        row_height = 80  # Height of each row (adjust as needed)

        # Add vertical lines for each minute
        for i in range(1441):  # 1441 to include the last minute
            line = QGraphicsRectItem(i * 1, -30, 1, 100 + row_height)
            line.setBrush(QColor(0, 0, 0))
            scene.addItem(line)

            # Add labels for each hour
            if i % 60 == 0:
                label = QGraphicsTextItem(f"{i // 60:02d}:00")
                label.setPos(i * 1 - 10, -70)
                font = QFont()
                font.setPointSize(15)
                label.setFont(font)
                scene.addItem(label)

        # Add Gantt items to the scene (specified in minutes)
        group: ForgererGroup
        for idx, group in schedule.items():
            oven: OvenSchedule
            for i, oven in enumerate(group.ovens_under_control):
                for task in oven.tasks:
                    scene.addItem(GanttItem(task.name, task.start_time, task.end_time - task.start_time, row_height, i + 1))

        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw Gantt diagram background (grid lines, labels, etc.)
        # You can customize this part based on your requirements.

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GanttDiagram()
    sys.exit(app.exec())
