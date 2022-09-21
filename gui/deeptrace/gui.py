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
from .utils import *

class DeepTraCE(QMainWindow):
    app = None
    def __init__(self,app = None):
        '''
        Graphical interface for the DeepTraCE pipeline.
        '''
        super(DeepTraCE,self).__init__()
        mainw = QWidget()
        self.setCentralWidget(mainw)
        lay = QHBoxLayout()
        mainw.setLayout(lay)
        self.analysis_steps = dict(trailmap=True)
        w = QGroupBox('TrailMap')
        l = QFormLayout()
        w.setLayout(l)
        run_trailmap = QCheckBox()
        run_trailmap.setChecked(self.analysis_steps['trailmap'])
        def tmap_update(val):
            self.analysis_steps['trailmap'] = not self.analysis_steps['trailmap']
        run_trailmap.stateChanged.connect(tmap_update)
        label = QLabel("Perform analysis: ")
        info = '''Run cell or axon discovery using a TrailMap model.'''
        label.setToolTip(info)
        run_trailmap.setToolTip(info)
        l.addRow(label,run_trailmap)
        lay.addWidget(w)
        
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

