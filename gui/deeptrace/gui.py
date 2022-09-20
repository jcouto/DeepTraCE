from PyQt5.QtWidgets import (QWidget,
                             QApplication,
                             QGridLayout,
                             QFormLayout,
                             QVBoxLayout,
                             QHBoxLayout,
                             QTabWidget,
                             QCheckBox,
                             QTextEdit,
                             QLineEdit,
                             QComboBox,
                             QSlider,
                             QPushButton,
                             QLabel,
                             QAction,
                             QWidgetAction,
                             QMenuBar,
                             QDoubleSpinBox,
                             QGraphicsView,
                             QGraphicsScene,
                             QGraphicsItem,
                             QGraphicsLineItem,
                             QGroupBox,
                             QTableWidget,
                             QMainWindow,
                             QDockWidget,
                             QFileDialog,
                             QDialog,
                             QInputDialog)
from PyQt5.QtGui import QImage, QPixmap,QBrush,QPen,QColor,QFont
from PyQt5.QtCore import Qt,QSize,QRectF,QLineF,QPointF,QTimer
import sys

class DeepTraCE(QMainWindow):
    app = None
    def __init__(self,app = None):
        '''
        Graphical interface for the DeepTraCE pipeline.
        '''
        super(DeepTraCE,self).__init__()
        self.show()
def main():
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    import os
    import json
    parser = ArgumentParser(description='DeepTraCE',formatter_class=RawDescriptionHelpFormatter)
    opts = parser.parse_args()
    app = QApplication(sys.argv)
    w = DeepTraCE(app = app)
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    main()

