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
                             QInputDialog,
                             QMessageBox,
                             QStyle)
from PyQt5.QtWidgets import QListWidget,QListWidgetItem,QShortcut
from PyQt5.QtGui import QImage, QPixmap,QBrush,QPen,QColor,QFont,QKeySequence
from PyQt5.QtCore import Qt,QSize,QRectF,QLineF,QPointF,QTimer,QDir

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
        self.setWindowTitle('DeepTraCE')
        self.setCentralWidget(mainw)
        lay = QGridLayout()
        mainw.setLayout(lay)
        self.experiments = []
        self.analysis_steps = dict(trailmap=True,
                                   rotate=False,
                                   angles = [0,0,0])
        self.downsampled_stack = None
        self.rotated_stack = None
        w = QGroupBox('TrailMap')
        l = QFormLayout()
        w.setLayout(l)
        run_trailmap = QCheckBox()
        run_trailmap.setChecked(self.analysis_steps['trailmap'])
        def tmap_update(val):
            self.analysis_steps['trailmap'] = not self.analysis_steps['trailmap']
        run_trailmap.stateChanged.connect(tmap_update)
        self.models = glob(pjoin(
            deeptrace_preferences['trailmap']['models_folder'],
            '*.hdf5')) 
        self.models_table = QListWidget()
        if not len(self.models):
            print('Models path is probably not correct? Check the preference file.')
            sys.exit(1)
        for m in self.models:
            self.models_table.addItem(os.path.basename(m))
        l.addRow(self.models_table)
        
        #label = QLabel("Analysis: ")
        #info = '''Run cell or axon discovery using a TrailMap model.'''
        #label.setToolTip(info)
        #run_trailmap.setToolTip(info)
        #l.addRow(label,run_trailmap)
        ww = QWidget()
        ll = QVBoxLayout()
        ww.setLayout(ll)
        self.table = QListWidget()
        add = QPushButton('Add experiment')
        icon = self.style().standardIcon(QStyle.SP_DirIcon)
        add.setIcon(icon)
        ll.addWidget(add)
        self.stack = None
        def addfolder():
            folder = QFileDialog.getExistingDirectory(ww,"Select 488nm data","")
            if not folder is None and not folder == '':
                folder = QDir.toNativeSeparators(folder)
                exp = dict(ch0=folder)
                folder = QFileDialog.getExistingDirectory(ww,"Select 640nm data","")
                if not folder is None and not folder == '':
                    folder = QDir.toNativeSeparators(folder)
                    exp['ch1'] = folder
                self.experiments.append(exp)
                self.table.insertItem(0,os.path.basename(exp['ch0']))
                print('Experiments:',self.experiments,flush = True)
                self.stack = BrainStack(**self.experiments[-1])
                self.stackim.setImage(self.stack[400])
                self.stackslider.setMaximum(len(self.stack)-1)
                self.stackslider.setMinimum(0)
                self.stackslider.setSingleStep(10)
                self.stackslider.setValue(400)
                self.downsampled_stack = None
                self.rotated_stack = None

        add.clicked.connect(addfolder)
        def clicked():
            item = self.table.currentRow()
            self.table.takeItem(item)
            self.experiments = [e for i,e in enumerate(self.experiments) if not i in [item]]
            print('Experiments:',self.experiments,flush = True)

        dd = QShortcut(QKeySequence(Qt.Key_Delete),self.table)
        dd.activated.connect(clicked)

        def load_plot(item):
            idx = self.table.currentRow()
            self.stack = BrainStack(**self.experiments[idx])
            print('Loaded stack {0}'.format(item.text()))
            self.stackim.setImage(self.stack[100])
            self.stackslider.setMaximum(len(self.stack)-1)
            self.stackslider.setMinimum(0)
            self.stackslider.setSingleStep(10)
            self.stackslider.setValue(400)
            self.downsampled_stack = None
            self.rotated_stack = None


        self.table.itemClicked.connect(load_plot) # connect itemClicked to Clicked method


        ll.addWidget(self.table)
        lay.addWidget(ww,1,0)
        lay.addWidget(w,2,0)
        
        run_button = QPushButton('Run analysis')
        lay.addWidget(run_button,0,0)
        run_button.clicked.connect(self.analyse)

        #rotation
        w = QGroupBox('Rotation')
        l = QFormLayout()
        w.setLayout(l)
        self.rotatew = QCheckBox()
        self.rotatew.setChecked(self.analysis_steps['rotate'])
        def rotate_sel(val):
            self.analysis_steps['rotate'] = not self.analysis_steps['rotate']
        self.rotatew.stateChanged.connect(rotate_sel)
        l.addRow(QLabel('Perform'),self.rotatew)
        labelx = QLabel('X [0]')
        labely = QLabel('Y [0]')
        labelz = QLabel('Z [0]')
        anglex = QSlider(Qt.Horizontal)
        angley = QSlider(Qt.Horizontal)
        anglez = QSlider(Qt.Horizontal)
        for a in [anglex,angley,anglez]:
            a.setMaximum(10*100)
            a.setMinimum(-10*100)
            a.setSingleStep(1)
        def move_x(val):
            self.analysis_steps['angles'][0] = val/100
            labelx.setText('X [{0:0.2f}]'.format(val/100))
        anglex.valueChanged.connect(move_x)
        def move_y(val):
            self.analysis_steps['angles'][1] = val/100
            labely.setText('Y [{0}]'.format(val/100))
        angley.valueChanged.connect(move_y)
        def move_z(val):
            self.analysis_steps['angles'][2] = val/100
            labelz.setText('Z [{0}]'.format(val/100))

        anglez.valueChanged.connect(move_z)

        l.addRow(labelx,anglex)
        l.addRow(labely,angley)
        l.addRow(labelz,anglez)
        def previewrotate():
            if not self.stack is None:
                nframes = 150
                val = self.stackslider.value()
                if self.downsampled_stack is None:
                    print('Downsampling sub-stack')
                    self.downsampled_stack = downsample_stack(self.stack[val:val+nframes],
                                                              scales = [0.40625,0.40625,0.3])
                print('Rotating stack')
                self.rotated_stack = rotate_stack(self.downsampled_stack,
                                                  anglex.value()/100,
                                                  angley.value()/100,
                                                  anglez.value()/100)
                im = self.rotated_stack[int(len(self.rotated_stack)/2)]
                impre =  self.downsampled_stack[int(len(self.rotated_stack)/2)]
                im = np.stack([im,impre,impre*0]).transpose([1,2,0])
                self.stackim.setImage(im)
                self.display_channels.setCurrentIndex(3)
                channels_update(3)
        preview_button = QPushButton('Preview')
        l.addRow(preview_button)
        preview_button.clicked.connect(previewrotate)

        lay.addWidget(w,3,0)
        
        import pyqtgraph as pg
        self.stackslider = QSlider(Qt.Horizontal)
        lay.addWidget(self.stackslider,0,1,1,2)
        def move_stack(val):
            if not self.stack is None:
                if self.display_channels.currentIndex()==3:
                    im = self.rotated_stack[self.stackslider.value()]
                    impre =  self.downsampled_stack[self.stackslider.value()]
                    im = np.stack([im,impre,impre*0]).transpose([1,2,0])
                else:
                    im = self.stack[val]
                self.stackim.setImage(im)
        
        self.stackslider.valueChanged.connect(move_stack)
        label = QLabel("Show channel: ")
        self.display_channels = QComboBox()
        self.display_channels.addItem('ch0')
        self.display_channels.addItem('ch1')
        self.display_channels.addItem('both')
        self.display_channels.addItem('rotation_preview')
        def channels_update(index):
            if hasattr(self.stack,'output_rgb'):
                if index < 2:
                    self.stack.output_rgb = False
                    self.stack.channel = index
                    im = self.stack[self.stackslider.value()]
                    self.stackslider.setMaximum(len(self.stack)-1)
                    self.stackslider.setMinimum(0)
                    self.stackslider.setSingleStep(10)
                elif index == 2:
                    self.stack.output_rgb = True
                    self.stackslider.setMaximum(len(self.stack)-1)
                    self.stackslider.setMinimum(0)
                    self.stackslider.setSingleStep(10)
                im = self.stack[self.stackslider.value()]
                if index == 3:
                    if not self.rotated_stack is None:
                        self.stackslider.setMaximum(len(self.rotated_stack)-1)
                        self.stackslider.setMinimum(0)
                        self.stackslider.setSingleStep(1)
                        self.stackslider.setValue(int(len(self.rotated_stack)/2))

                        im = self.rotated_stack[self.stackslider.value()]
                        impre =  self.downsampled_stack[self.stackslider.value()]
                        im = np.stack([im,impre,impre*0]).transpose([1,2,0])
            self.stackim.setImage(im)
        self.display_channels.activated.connect(channels_update)
        lay.addWidget(self.display_channels,0,3,1,1)
        #run_trailmap.setToolTip(info)
        #l.addRow(label,run_trailmap)

        self.stackwin = pg.GraphicsLayoutWidget()
        lay.addWidget(self.stackwin,1,1,3,6)
        self.stackwin.setMinimumWidth(800)
        self.stackpl = self.stackwin.addPlot()
        self.stackpl.getViewBox().invertY(True)
        self.stackpl.getViewBox().setAspectLocked(False)
        
        self.stackim = pg.ImageItem()
        self.stackwin.setCentralWidget(self.stackpl)
        self.stackpl.addItem(self.stackim)
        
        self.stackpl.setClipToView(True)
        self.show()
    
    def analyse(self):
        code_path = deeptrace_preferences['trailmap']['path']
        trailmap_env = deeptrace_preferences['trailmap']['environment']
        modelidx = self.models_table.currentRow()
        if modelidx is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('Select a model before running TRAILMAP')
            msg.setWindowTitle("SELECT A MODEL... PLEASE...")
            msg.exec_()
            return
        model_path = pjoin(deeptrace_preferences['trailmap']['models_folder'],
                           self.models[modelidx])
        for exp in self.experiments:
            input_folders = []
            if not 'ch1' in exp.keys():
                print("No channel 2 for experiment: " + exp['ch0'])
                continue
            input_folders.append(exp['ch1'])
            #1 run trailmap
            # in the future check of results are already there
            output_name = "{0}_seg-".format(
                os.path.splitext(os.path.basename(model_path))[0]) + os.path.basename(input_folders[0])
            output_dir = os.path.dirname(input_folders[0])
            output_folder = os.path.join(output_dir, output_name)
            print('TRAILMAP OUTPUT: {0}'.format(output_folder))
            if not os.path.exists(output_folder):
                run_segment_brain_on_model(code_path, model_path, input_folders, trailmap_env)
            else:
                print('TRAILMAP analysis was already done!')
            #2 downsample the autofluorescence
            stack = BrainStack(exp['ch0'])
            stack10um = downsample_stack(stack,scales = [0.40625,0.40625,0.3])
            imsave(pjoin(exp['ch0']+'_scaled','10um.tif'),stack10um)
            self.stack = stack10um
            self.stackslider.setMaximum(int(self.stack.shape))
            print('Saved downsampled stack.')
            #3 downsample the trailmap result
            
            #4 rotate these both
            
            # run elastix

def run_segment_brain_on_model(code_path, model_path, input_folders,trailmap_env):
    code = '''
import os
import sys
sys.path.append('{code_path}')
from inference import *
from models import *
import shutil

if __name__ == "__main__":
    input_batch = sys.argv[1:]
    # Verify each path is a directory
    for input_folder in input_batch:
        if not os.path.isdir(input_folder):
            raise Exception(input_folder + " is not a directory. Inputs must be a folder of files. Please refer to readme for more info")
    # Load the network
    weights_path = '{model_path}'

    model = get_net()
    model.load_weights(weights_path)

    for input_folder in input_batch:
        # Remove trailing slashes
        input_folder = os.path.normpath(input_folder)
        # Output folder name
        output_name = "{model}_seg-" + os.path.basename(input_folder)
        output_dir = os.path.dirname(input_folder)
        output_folder = os.path.join(output_dir, output_name)
        # Create output directory. Overwrite if the directory exists
        if os.path.exists(output_folder):
            print(output_folder + " already exists. Will be overwritten")
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        # Segment the brain
        print('The results will be stored in:'+output_f)
        segment_brain(input_folder, output_folder, model)
''' # this is from trailmap, if the environment could be the same (imagej would work) we could skip this

    tmpf = pjoin(deeptrace_path,'run_trailmap.py')
    with open(tmpf,'w') as fd:
        fd.write(code.format(model_path=model_path,
                             model=os.path.splitext(os.path.basename(model_path))[0],
                             code_path = code_path))
    import subprocess as sub
    cmd = r'cd {0} & conda activate {1} & python run_trailmap.py {2}'.format(
        deeptrace_path, trailmap_env, ' '.join(input_folders))
    print(cmd)
    sub.call(cmd,shell = True)

    
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

