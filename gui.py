import sys

from pydicom import dcmread
from PIL import ImageTk, Image, ImageQt

import numpy as np

import matplotlib.animation as plt_animation
import matplotlib as plt
import matplotlib.pyplot as plt_py
from pydicom.fileset import FileSet
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer,QDateTime, Qt, Slot,QPointF, Signal, QRect
from PySide6.QtGui import QAction, QPainter,QImage, QPixmap, QPen, QIntValidator, QValidator, QColor, QKeySequence, QPolygon, QPolygonF, QBrush, QMouseEvent, QFont, QCursor
from PySide6.QtWidgets import (QApplication, QLabel,QMainWindow, QPushButton, QWidget, QListWidget,
                               QLineEdit,QFileDialog,QVBoxLayout, QHBoxLayout,QDialogButtonBox,
                               QGridLayout, QCheckBox, QMessageBox, QSizePolicy, QSlider, QGraphicsView, 
                               QGraphicsScene, QStyle,QSpacerItem, QRadioButton, QTabWidget,QDialog)
from random import randint, choice
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import axes3d
import matplotlib.patches as mpatches
import pyqtgraph as pg
from scipy.interpolate import CubicSpline, UnivariateSpline, LSQUnivariateSpline
import keyboard

from leaflet_graph import Leaflet_Graph
from model3D_tab import Model_3D_Tab
from mpr2D_tab import MPR_2D_Tab
from view_tabs import View_tabs
from excel_writer import Excel_Writer
import pathlib
import os



class MyPopup(QDialog):
    def __init__(self, excel_writer, my_gui):
        QDialog.__init__(self)
        self.my_gui = my_gui
        self.excel_writer = excel_writer
        self.dataloader = my_gui.dataloader
        self.valve_model = 'complete'
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

                            # User,pre:mv,   ant,  post,   raw,   dat, spline,post:mv,  ant,  post,  raw, dat, spline
        self.filled_fields= [False, False, False, False, False, False, False, False, False, False, False, False, False]

        self.user_field = QLineEdit()
        self.user_field.setText(str(''))
        self.user_field.setMaxLength(10)
        self.user_field.setAlignment(Qt.AlignLeft)
        self.user_field_label = QLabel("User: ")

        self.round_field = QLineEdit()
        self.round_field.setText(str(''))
        self.round_field.setValidator(QIntValidator())
        self.round_field.setMaxLength(2)
        self.round_field.setAlignment(Qt.AlignLeft)
        self.round_field_label = QLabel("Round: ")


        button_layout = QGridLayout()
        button_layout.addWidget(self.user_field_label,0,0)
        button_layout.addWidget(self.user_field,0,1)
        button_layout.addWidget(self.round_field_label,0,2)
        button_layout.addWidget(self.round_field,0,3)

        self.user_field.textChanged.connect(self.on_text_changed_user_field)
        self.user_field.textChanged.connect(self.on_text_changed_round_field)

        
        self.radio_button_layout = QHBoxLayout()
        self.complete_model_radio_button = QRadioButton("complete model")
        self.complete_model_radio_button.setChecked(True)
        self.complete_model_radio_button.toggled.connect(self.complete_model_selected)
        self.partial_model_radio_button = QRadioButton("partial model")
        self.partial_model_radio_button.setChecked(False)
        self.partial_model_radio_button.toggled.connect(self.partial_model_selected)
        self.no_model_radio_button = QRadioButton("no model")
        self.no_model_radio_button.setChecked(False)
        self.no_model_radio_button.toggled.connect(self.no_model_selected)

        self.radio_button_layout.addWidget(self.complete_model_radio_button)
        self.radio_button_layout.addWidget(self.partial_model_radio_button)
        self.radio_button_layout.addWidget(self.no_model_radio_button)


        self.file_lab_mv_pre = QLabel("Choose pre mv valve:") 
        self.file_text_mv_pre = QLineEdit(readOnly=True)
        self.file_button_mv_pre = QPushButton("Select file")
        self.file_lab_ant_pre = QLabel("Choose pre ant leaflet:") 
        self.file_text_ant_pre = QLineEdit(readOnly=True)
        self.file_button_ant_pre = QPushButton("Select file")
        self.file_lab_post_pre = QLabel("Choose pre post leaflet:") 
        self.file_text_post_pre = QLineEdit(readOnly=True)
        self.file_button_post_pre = QPushButton("Select file")

        self.file_lab_raw_pre = QLabel("Choose pre raw file:") 
        self.file_text_raw_pre = QLineEdit(readOnly=True)
        self.file_button_raw_pre = QPushButton("Select file")
        self.file_lab_dat_pre = QLabel("Choose pre dat file:") 
        self.file_text_dat_pre = QLineEdit(readOnly=True)
        self.file_button_dat_pre = QPushButton("Select file")


        self.file_lab_mv_intra = QLabel("Choose intra mv valve:") 
        self.file_text_mv_intra = QLineEdit(readOnly=True)
        self.file_button_mv_intra = QPushButton("Select file")
        self.file_lab_ant_intra = QLabel("Choose intra ant leaflet:") 
        self.file_text_ant_intra = QLineEdit(readOnly=True)
        self.file_button_ant_intra = QPushButton("Select file")
        self.file_lab_post_intra = QLabel("Choose intra post leaflet:") 
        self.file_text_post_intra = QLineEdit(readOnly=True)
        self.file_button_post_intra = QPushButton("Select file")

        self.file_lab_raw_intra = QLabel("Choose intra raw file:") 
        self.file_text_raw_intra = QLineEdit(readOnly=True)
        self.file_button_raw_intra = QPushButton("Select file")
        self.file_lab_dat_intra = QLabel("Choose intra dat file:") 
        self.file_text_dat_intra = QLineEdit(readOnly=True)
        self.file_button_dat_intra = QPushButton("Select file")

        self.file_lab_spline_model_pre = QLabel("Choose spline model pre file:") 
        self.file_text_spline_model_pre = QLineEdit(readOnly=True)
        self.file_button_spline_model_pre = QPushButton("Select file")
        self.file_lab_spline_model_intra = QLabel("Choose spline model intra file:") 
        self.file_text_spline_model_intra = QLineEdit(readOnly=True)
        self.file_button_spline_model_intra = QPushButton("Select file")

        self.file_lab_apex_pre = QLabel("Choose apex and clip pre file:") 
        self.file_text_apex_pre = QLineEdit(readOnly=True)
        self.file_button_apex_pre = QPushButton("Select file")
        self.file_lab_apex_intra = QLabel("Choose apex and clip intra file:") 
        self.file_text_apex_intra = QLineEdit(readOnly=True)
        self.file_button_apex_intra = QPushButton("Select file")

        self.file_lab_annulus_pre = QLabel("Choose annulus pre file:") 
        self.file_text_annulus_pre = QLineEdit(readOnly=True)
        self.file_button_annulus_pre = QPushButton("Select file")
        self.file_lab_annulus_intra = QLabel("Choose annulus intra file:") 
        self.file_text_annulus_intra = QLineEdit(readOnly=True)
        self.file_button_annulus_intra = QPushButton("Select file")



        self.file_button_mv_pre.clicked.connect(lambda: self.set_filepath('mv', 'pre', self.valve_model))
        self.file_button_ant_pre.clicked.connect(lambda: self.set_filepath('ant', 'pre', self.valve_model))
        self.file_button_post_pre.clicked.connect(lambda: self.set_filepath( 'post', 'pre', self.valve_model))

        self.file_button_raw_pre.clicked.connect(lambda: self.set_filepath('raw', 'pre', self.valve_model))
        self.file_button_dat_pre.clicked.connect(lambda: self.set_filepath('dat', 'pre', self.valve_model))

        self.file_button_mv_intra.clicked.connect(lambda: self.set_filepath('mv', 'intra', self.valve_model))
        self.file_button_ant_intra.clicked.connect(lambda: self.set_filepath('ant', 'intra', self.valve_model))
        self.file_button_post_intra.clicked.connect(lambda: self.set_filepath( 'post', 'intra', self.valve_model))

        self.file_button_raw_intra.clicked.connect(lambda: self.set_filepath('raw', 'intra', self.valve_model))
        self.file_button_dat_intra.clicked.connect(lambda: self.set_filepath('dat', 'intra', self.valve_model))

        self.file_button_spline_model_pre.clicked.connect(lambda: self.set_filepath('spline', 'pre', self.valve_model))
        self.file_button_spline_model_intra.clicked.connect(lambda: self.set_filepath('spline', 'intra', self.valve_model))

        self.file_button_apex_pre.clicked.connect(lambda: self.set_filepath('apex', 'pre', self.valve_model))
        self.file_button_apex_intra.clicked.connect(lambda: self.set_filepath('apex', 'intra', self.valve_model))

        self.file_button_annulus_pre.clicked.connect(lambda: self.set_filepath('annulus', 'pre', self.valve_model))
        self.file_button_annulus_intra.clicked.connect(lambda: self.set_filepath('annulus', 'intra', self.valve_model))

        self.complete_model_selected()

        # load file
        load_layout = QGridLayout()
        load_layout.addWidget(self.file_lab_mv_pre,0,0)
        load_layout.addWidget(self.file_text_mv_pre,0,1)
        load_layout.addWidget(self.file_button_mv_pre,0,2)
        load_layout.addWidget(self.file_lab_ant_pre, 1,0)
        load_layout.addWidget(self.file_text_ant_pre,1,1)
        load_layout.addWidget(self.file_button_ant_pre, 1,2)
        load_layout.addWidget(self.file_lab_post_pre,2,0)
        load_layout.addWidget(self.file_text_post_pre,2,1)
        load_layout.addWidget(self.file_button_post_pre,2,2)

        load_layout.addWidget(self.file_lab_raw_pre, 3,0)
        load_layout.addWidget(self.file_text_raw_pre,3,1)
        load_layout.addWidget(self.file_button_raw_pre, 3,2)
        load_layout.addWidget(self.file_lab_dat_pre,4,0)
        load_layout.addWidget(self.file_text_dat_pre,4,1)
        load_layout.addWidget(self.file_button_dat_pre,4,2)

        load_layout.addWidget(self.file_lab_mv_intra , 0, 3)
        load_layout.addWidget(self.file_text_mv_intra , 0, 4)
        load_layout.addWidget(self.file_button_mv_intra , 0, 5)
        load_layout.addWidget(self.file_lab_ant_intra, 1, 3)
        load_layout.addWidget(self.file_text_ant_intra,1, 4)
        load_layout.addWidget(self.file_button_ant_intra, 1, 5)
        load_layout.addWidget(self.file_lab_post_intra,2, 3)
        load_layout.addWidget(self.file_text_post_intra,2, 4)
        load_layout.addWidget(self.file_button_post_intra,2, 5)

        load_layout.addWidget(self.file_lab_raw_intra, 3, 3)
        load_layout.addWidget(self.file_text_raw_intra,3, 4)
        load_layout.addWidget(self.file_button_raw_intra, 3, 5)
        load_layout.addWidget(self.file_lab_dat_intra,4, 3)
        load_layout.addWidget(self.file_text_dat_intra,4, 4)
        load_layout.addWidget(self.file_button_dat_intra,4, 5)

        load_layout.addWidget(self.file_lab_spline_model_pre, 5, 0)
        load_layout.addWidget(self.file_text_spline_model_pre,5, 1)
        load_layout.addWidget(self.file_button_spline_model_pre, 5, 2)
        load_layout.addWidget(self.file_lab_spline_model_intra,5, 3)
        load_layout.addWidget(self.file_text_spline_model_intra,5, 4)
        load_layout.addWidget(self.file_button_spline_model_intra,5, 5)

        load_layout.addWidget(self.file_lab_apex_pre,6, 0)
        load_layout.addWidget(self.file_text_apex_pre,6, 1)
        load_layout.addWidget(self.file_button_apex_pre,6, 2)
        load_layout.addWidget(self.file_lab_apex_intra,6, 3)
        load_layout.addWidget(self.file_text_apex_intra,6, 4)
        load_layout.addWidget(self.file_button_apex_intra,6, 5)

        load_layout.addWidget(self.file_lab_annulus_pre,7, 0)
        load_layout.addWidget(self.file_text_annulus_pre,7, 1)
        load_layout.addWidget(self.file_button_annulus_pre,7, 2)
        load_layout.addWidget(self.file_lab_annulus_intra,7, 3)
        load_layout.addWidget(self.file_text_annulus_intra,7, 4)
        load_layout.addWidget(self.file_button_annulus_intra,7, 5)

        start_button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Measurements")
        self.start_button.setEnabled(False)
        start_button_layout.addWidget(self.start_button,0)

        self.start_button.clicked.connect(self.start_button_was_pressed)

        window_layout = QGridLayout()
        window_layout.addLayout(button_layout, 0,0)
        window_layout.addLayout(self.radio_button_layout, 1, 0)
        window_layout.addLayout(load_layout, 2,0)
        window_layout.addLayout(start_button_layout, 3,0)
        
        self.setLayout(window_layout)
    
    def start_button_was_pressed(self):
       
        self.close()
        self.my_gui.init_rest_of_gui()


    def on_text_changed_user_field(self):
        if self.user_field.text() != '':
            self.filled_fields[0] = True 
        if np.all(self.filled_fields) == True:
            self.start_button.setEnabled(True)
        self.excel_writer.add_value('pre', 'observer', self.user_field.text())
        self.excel_writer.add_value('intra', 'observer', self.user_field.text())

    def on_text_changed_round_field(self):
        self.excel_writer.add_value('pre', 'round', self.round_field.text())
        self.excel_writer.add_value('intra', 'round', self.round_field.text())


    def on_text_changed_apex_pre_field(self):
        self.dataloader.apex_clip_pre_path = None

    def on_text_changed_apex_intra_field(self):
        apex_intra = self.check_array_type(self.apex_intra_field.text())
        if apex_intra is not None:
            self.dataloader.apex_intra= apex_intra
            self.excel_writer.add_value('intra', 'apex', apex_intra)
        else:
            self.excel_writer.add_value('intra', 'apex', 'None')


    def check_array_type(self, var):
        if isinstance(var, str):
            try:
                var = eval(var)  # parse the string as a Python expression
            except SyntaxError:
                print( "Invalid string format")
                return None
            except NameError:
                print( "Invalid string format")
                return None
        if isinstance(var, (np.ndarray, list, tuple)):
            if len(var) == 3:
                if all(isinstance(x, (int, float)) for x in var):
                    print( "Array of 3 int or float values")
                    return var
                else:
                    print( "Array of 3 values, but not all int or float")
                    return None
            else:
                print( "Array, but not of length 3")
                return None
        else:
            print( "Not an array")
            return None

    def set_filepath(self, mode, pre_or_intra, valve_model):
        """Choose and set file via FileDialog. Emit Display signal."""
        #mode = inp, raw, dat
        if mode == 'mv' and pre_or_intra == 'pre':
            file_text = self.file_text_mv_pre
            filter = "inp files (*.inp)" 
        elif mode == 'mv' and pre_or_intra == 'intra':
            file_text = self.file_text_mv_intra
            filter = "inp files (*.inp)" 

        elif mode == 'ant' and pre_or_intra == 'pre':
            file_text = self.file_text_ant_pre
            filter = "inp files (*.inp)" 
        elif mode == 'ant' and pre_or_intra == 'intra':
            file_text = self.file_text_ant_intra
            filter = "inp files (*.inp)" 

        elif mode == 'post' and pre_or_intra == 'pre':
            file_text = self.file_text_post_pre
            filter = "inp files (*.inp)" 
        elif mode == 'post' and pre_or_intra == 'intra':
            file_text = self.file_text_post_intra
            filter = "inp files (*.inp)" 

        elif mode == 'raw' and pre_or_intra == 'pre':
            file_text = self.file_text_raw_pre
            filter = "Raw Files (*.raw)"
        elif mode == 'raw' and pre_or_intra == 'intra':
            file_text = self.file_text_raw_intra
            filter = "Raw Files (*.raw)"
        
        elif mode == 'dat' and pre_or_intra == 'pre':
            file_text = self.file_text_dat_pre
            filter = "Dat Files (*.dat)"     
        elif mode == 'dat' and pre_or_intra == 'intra':
            file_text = self.file_text_dat_intra
            filter = "Dat Files (*.dat)"   
            
        elif mode == 'spline' and pre_or_intra == 'pre':
            file_text = self.file_text_spline_model_pre
            filter = "MAT files (*.mat)"      
        elif mode == 'spline' and pre_or_intra == 'intra':
            file_text = self.file_text_spline_model_intra
            filter = "MAT files (*.mat)"     
            
        elif mode =='apex' and pre_or_intra == 'pre':
            file_text = self.file_text_apex_pre
            filter = "Text files (*.txt)"  
        elif mode =='apex' and pre_or_intra == 'intra':
            file_text = self.file_text_apex_intra
            filter = "Text files (*.txt)"    
            
        elif mode =='annulus' and pre_or_intra == 'pre':
            file_text = self.file_text_annulus_pre
            filter = "Text files (*.txt)"  
        elif mode =='annulus' and pre_or_intra == 'intra':
            file_text = self.file_text_annulus_intra
            filter = "Text files (*.txt)"                                              #set the filter to just show h5 data

        with open(str(pathlib.Path(__file__).parent.resolve())+'/last_path.txt') as f:      #open the last_path file and read the last used path
            lines = f.readlines()

        f.close

        try:
            last_path=str(lines[0])
        except IndexError:                                                                  #if there is no line in the last_path file go to /home/Users/
            last_path="/home/Users/"

        
        file = QFileDialog.getOpenFileName(self,"Open File", last_path,filter)[0]        #open the file dialog 
        #print("file h5")

        way=os.path.dirname(file)                                                        #get the path of the selected file
        

        if len(way)!=0:
            with open(str(pathlib.Path(__file__).parent.resolve())+'/last_path.txt', 'w') as f: #save the path of the last used file and save it in the last_path file
                line1 = str(way)
                f.writelines(([line1]))

        f.close

        
        file_text.setText(file)

    


        if mode == 'mv':
            if pre_or_intra == 'pre':
                self.filled_fields[1] = True 
                self.dataloader.mv_model_pre_path = file
            elif pre_or_intra == 'intra':
                self.filled_fields[7] = True 
                self.dataloader.mv_model_intra_path = file
            if valve_model == 'complete':
                self.excel_writer.add_value(pre_or_intra, 'mv_model_complete', file)
            elif valve_model == 'partial':
                self.excel_writer.add_value(pre_or_intra, 'mv_model_partial', file)
        elif mode == 'ant':
            if pre_or_intra == 'pre':
                self.filled_fields[2] = True 
                self.dataloader.ant_model_pre_path= file
            elif pre_or_intra == 'intra':
                self.filled_fields[8] = True 
                self.dataloader.ant_model_intra_path= file
            if valve_model == 'complete':
                self.excel_writer.add_value(pre_or_intra, 'ant_model_complete', file)
            elif valve_model == 'partial':
                self.excel_writer.add_value(pre_or_intra, 'ant_model_partial', file)
        elif mode == 'post':
            if pre_or_intra == 'pre':
                self.filled_fields[3] = True 
                self.dataloader.post_model_pre_path= file
            elif pre_or_intra == 'intra':
                self.filled_fields[9] = True 
                self.dataloader.post_model_intra_path= file
            if valve_model == 'complete':
                self.excel_writer.add_value(pre_or_intra, 'post_model_complete', file)
            elif valve_model == 'partial':
                self.excel_writer.add_value(pre_or_intra, 'post_model_partial', file)
        elif mode == 'raw':
            if pre_or_intra == 'pre':
                self.filled_fields[4] = True 
                self.dataloader.raw_data_pre_path= file
            elif pre_or_intra == 'intra':
                self.filled_fields[10] = True 
                self.dataloader.raw_data_intra_path= file
            self.excel_writer.add_value(pre_or_intra, 'raw_data', file)
        elif mode == 'dat':
            if pre_or_intra == 'pre':
                self.filled_fields[5] = True 
                self.dataloader.dat_data_pre_path= file
            elif pre_or_intra == 'intra':
                self.filled_fields[11] = True   
                self.dataloader.dat_data_intra_path= file
            self.excel_writer.add_value(pre_or_intra, 'dat_data', file)
        elif mode == 'spline':
            if pre_or_intra == 'pre':
                self.filled_fields[6] = True 
                self.dataloader.spline_model_pre_path= file
            elif pre_or_intra == 'intra':
                self.filled_fields[12] = True 
                self.dataloader.spline_model_intra_path= file  
            if valve_model == 'complete':
                self.excel_writer.add_value(pre_or_intra, 'spline_model_complete', file)
            elif valve_model == 'partial':
                self.excel_writer.add_value(pre_or_intra, 'spline_model_partial', file)
        elif mode == 'apex':
            if pre_or_intra == 'pre':
                self.dataloader.apex_clip_pre_path = file
            elif pre_or_intra == 'intra': 
                self.dataloader.apex_clip_intra_path = file
        elif mode == 'annulus':
            if pre_or_intra == 'pre':
                self.dataloader.annulus_pre_path = file
            elif pre_or_intra == 'intra': 
                self.dataloader.annulus_intra_path = file
        if np.all(self.filled_fields) == True:
            self.start_button.setEnabled(True)
        return
    
    def complete_model_selected(self):
        self.valve_model = 'complete'

        self.file_lab_mv_pre.setVisible(True)
        self.file_text_mv_pre.setVisible(True)
        self.file_button_mv_pre.setVisible(True)
        self.file_lab_ant_pre.setVisible(True)
        self.file_text_ant_pre.setVisible(True)
        self.file_button_ant_pre.setVisible(True)
        self.file_lab_post_pre.setVisible(True)
        self.file_text_post_pre.setVisible(True)
        self.file_button_post_pre.setVisible(True)

        self.file_lab_mv_intra.setVisible(True)
        self.file_text_mv_intra.setVisible(True)
        self.file_button_mv_intra.setVisible(True)
        self.file_lab_ant_intra.setVisible(True)
        self.file_text_ant_intra.setVisible(True)
        self.file_button_ant_intra.setVisible(True)
        self.file_lab_post_intra.setVisible(True)
        self.file_text_post_intra.setVisible(True)
        self.file_button_post_intra.setVisible(True)

        self.file_lab_spline_model_pre.setVisible(True)
        self.file_text_spline_model_pre.setVisible(True)
        self.file_button_spline_model_pre.setVisible(True)
        self.file_lab_spline_model_intra.setVisible(True)
        self.file_text_spline_model_intra.setVisible(True)
        self.file_button_spline_model_intra.setVisible(True)

        self.file_lab_annulus_pre.setVisible(False)
        self.file_text_annulus_pre.setVisible(False)
        self.file_button_annulus_pre.setVisible(False)
        self.file_lab_annulus_intra.setVisible(False)
        self.file_text_annulus_intra.setVisible(False)
        self.file_button_annulus_intra.setVisible(False)


    def partial_model_selected(self):
        self.valve_model = 'partial'

        self.file_lab_mv_pre.setVisible(True)
        self.file_text_mv_pre.setVisible(True)
        self.file_button_mv_pre.setVisible(True)
        self.file_lab_ant_pre.setVisible(True)
        self.file_text_ant_pre.setVisible(True)
        self.file_button_ant_pre.setVisible(True)
        self.file_lab_post_pre.setVisible(True)
        self.file_text_post_pre.setVisible(True)
        self.file_button_post_pre.setVisible(True)

        self.file_lab_mv_intra.setVisible(True)
        self.file_text_mv_intra.setVisible(True)
        self.file_button_mv_intra.setVisible(True)
        self.file_lab_ant_intra.setVisible(True)
        self.file_text_ant_intra.setVisible(True)
        self.file_button_ant_intra.setVisible(True)
        self.file_lab_post_intra.setVisible(True)
        self.file_text_post_intra.setVisible(True)
        self.file_button_post_intra.setVisible(True)

        self.file_lab_spline_model_pre.setVisible(True)
        self.file_text_spline_model_pre.setVisible(True)
        self.file_button_spline_model_pre.setVisible(True)
        self.file_lab_spline_model_intra.setVisible(True)
        self.file_text_spline_model_intra.setVisible(True)
        self.file_button_spline_model_intra.setVisible(True)

        self.file_lab_annulus_pre.setVisible(False)
        self.file_text_annulus_pre.setVisible(False)
        self.file_button_annulus_pre.setVisible(False)
        self.file_lab_annulus_intra.setVisible(False)
        self.file_text_annulus_intra.setVisible(False)
        self.file_button_annulus_intra.setVisible(False)


    def no_model_selected(self):
        self.valve_model = 'no'

        self.file_lab_mv_pre.setVisible(False)
        self.file_text_mv_pre.setVisible(False)
        self.file_button_mv_pre.setVisible(False)
        self.file_lab_ant_pre.setVisible(False)
        self.file_text_ant_pre.setVisible(False)
        self.file_button_ant_pre.setVisible(False)
        self.file_lab_post_pre.setVisible(False)
        self.file_text_post_pre.setVisible(False)
        self.file_button_post_pre.setVisible(False)

        self.file_lab_mv_intra.setVisible(False)
        self.file_text_mv_intra.setVisible(False)
        self.file_button_mv_intra.setVisible(False)
        self.file_lab_ant_intra.setVisible(False)
        self.file_text_ant_intra.setVisible(False)
        self.file_button_ant_intra.setVisible(False)
        self.file_lab_post_intra.setVisible(False)
        self.file_text_post_intra.setVisible(False)
        self.file_button_post_intra.setVisible(False)

        self.file_lab_spline_model_pre.setVisible(False)
        self.file_text_spline_model_pre.setVisible(False)
        self.file_button_spline_model_pre.setVisible(False)
        self.file_lab_spline_model_intra.setVisible(False)
        self.file_text_spline_model_intra.setVisible(False)
        self.file_button_spline_model_intra.setVisible(False)

        self.file_lab_annulus_pre.setVisible(True)
        self.file_text_annulus_pre.setVisible(True)
        self.file_button_annulus_pre.setVisible(True)
        self.file_lab_annulus_intra.setVisible(True)
        self.file_text_annulus_intra.setVisible(True)
        self.file_button_annulus_intra.setVisible(True)
    


class Gui(QMainWindow):

    
    def __init__(self, animation, dataloader, parent=None):
        super().__init__()
        self._main = QWidget()
        self.setCentralWidget(self._main)

        self.excel_writer = Excel_Writer('Bella', 'No_Mesh')

        self.animation = animation
        self.dataloader = dataloader
        self.frame = self.animation.frame


        self.show_popup()
        

    def init_rest_of_gui(self):

        all_files = self.dataloader.prepare_all_files()
        
        #The frames are not changed in sequence when the application is opened
        self.play_state = False

        self.draw_leaflet =False
        self.label1_set = False
        self.label2_set = False

        self.select_pre_leaflet = True

        # Main menu bar
        self.menu = self.menuBar()
        self.menu_file = self.menu.addMenu("File")
        exit = QAction("Exit", self, triggered=qApp.quit)
        self.menu_file.addAction(exit)
        self.menu_about = self.menu.addMenu("&About")
        about = QAction("About Qt", self, shortcut=QKeySequence(QKeySequence.HelpContents),
                        triggered=qApp.aboutQt)
        self.menu_about.addAction(about)

        save_excel_sheet = QAction("Save excel sheet", self, triggered=self.excel_writer.save_excel_sheet)
        self.menu_file.addAction(save_excel_sheet)


        #Layouts - hold the sigle elements in the Gui
        main_layout = QVBoxLayout(self._main)


        self.pix = QPixmap()
        self.animation.pix = self.pix
        self.image_layout = QHBoxLayout()
        self.image_label = QLabel()
        self.image_layout.addWidget(self.image_label)

        
        self.volume_pre_op = all_files[0]
        self.volume_intra_op = all_files[1]
        
        self.tabs = View_tabs(self.volume_pre_op, self.volume_intra_op, #self.volume_pre_op, self.volume_intra_op,
                            all_files[2], all_files[3], all_files[4], #self.model_mv_pre, self.model_ant_pre, self.model_post_pre,
                            all_files[5], all_files[6], all_files[7], #self.model_mv_intra, self.model_ant_intra, self.model_post_intra,
                            all_files[8], all_files[9], #self.slice_thickness_pre, self.slice_thickness_intra, 
                            all_files[10], all_files[11], #self.spline_model_pre, self.spline_model_intra
                            all_files[12], all_files[13], #self.apex_pre, self.apex_intra
                            all_files[14], all_files[15], #self.clip_start_pre (None) self.clip_start_intra
                            all_files[16], all_files[17], #self.clip_end_pre (None) self.clip_end_intra
                            all_files[18], all_files[19], #self.annulus_pre_data, self.annulus_intra_data
                            self.excel_writer, self.dataloader)
        
        self.tabs.tab_thin_spline_model.animation = self.animation
        self.animation.mpr_image = self.get_mpr_image(self.tabs.tab_2D)


        #connecting the buttons to the according functions
        self.connect_buttons(self.tabs.tab_2D)
        self.connect_buttons(self.tabs.tab_2D_bent)
        
      
        main_layout.addWidget(self.tabs)



        #self.animation.image_label = self.image_label
        self.animation.image_label = self.image_layout
        #displays image when application is opened
        self.animation.update_figure() 
        #connects the timer to the function that plays the image sequence
        self.animation.timer.timeout.connect(self.animation.start_frame_sequence) 

        self.set_buttons_state((True, True, True, True, False), self.tabs.tab_2D)
        self.set_buttons_state((True, True, True, True, False), self.tabs.tab_2D_bent)
        

        self.show()
        self.tabs.tab_3D.mpr_slicer.prepare_visualization(True)

        

    def show_popup(self):
        # Create an instance of MyPopup
        self.popup = MyPopup(self.excel_writer, self)
        # Set the popup to be a modal dialog (optional)
        self.popup.setWindowModality(Qt.ApplicationModal)
        # Show the popup
        self.popup.show()

    def connect_buttons(self, page):
        page.play_button.clicked.connect(lambda: self.play_button_was_pressed(page))
        page.play_button.clicked.connect(self.animation.frame_animation_timer)
        page.last_button.clicked.connect(lambda: self.last_button_was_pressed(page))
        page.next_button.clicked.connect(lambda: self.next_button_was_pressed(page))
        page.select_leaflet_button.clicked.connect(lambda: self.select_new_leaflet_button_was_pressed(page))
        page.select_leaflet_done_button.clicked.connect(lambda: self.select_leaflet_done_button_was_pressed(page))
        page.select_pre_leaflet_button.clicked.connect(lambda: self.select_pre_leaflet_button_was_pressed(page))
        page.delete_selected_spline_button.clicked.connect(lambda: self.delete_selected_spline_button_was_pressed(page))

        #Might fire twice if not disconnected first
        try:
            self.tabs.tab_2D.leaflet_pre_plot.sceneObj.sigMouseClicked.disconnect()
            self.tabs.tab_2D.leaflet_intra_plot.sceneObj.sigMouseClicked.disconnect()
            self.tabs.tab_2D_bent.leaflet_pre_plot.sceneObj.sigMouseClicked.disconnect()
            self.tabs.tab_2D_bent.leaflet_intra_plot.sceneObj.sigMouseClicked.disconnect()
        except:
            pass

        self.tabs.tab_2D.leaflet_pre_plot.sceneObj.sigMouseClicked.connect(self.mouse_clicked)
        self.tabs.tab_2D.leaflet_intra_plot.sceneObj.sigMouseClicked.connect(self.mouse_clicked)
        self.tabs.tab_2D_bent.leaflet_pre_plot.sceneObj.sigMouseClicked.connect(self.mouse_clicked_bent)
        self.tabs.tab_2D_bent.leaflet_intra_plot.sceneObj.sigMouseClicked.connect(self.mouse_clicked_bent)



    def get_mpr_image(self, page):
        return page.mpr_pre_image
         

    def set_buttons_state(self, states, page):
        page.play_button.setEnabled(states[0])
        page.last_button.setEnabled(states[1])
        page.next_button.setEnabled(states[2])
        page.select_leaflet_button.setEnabled(states[3])
        page.select_leaflet_done_button.setEnabled(states[4])

    def play_button_was_pressed(self, page):
        self.play_state = not self.play_state
        if self.play_state == True:
            page.play_button.setText("Stop")
        else:
            page.play_button.setText("Play")

    def last_button_was_pressed(self, page):
        self.animation.go_to_last_frame()
        page.play_button.setText("Play")
        self.play_state = False

    def next_button_was_pressed(self, page):
        self.animation.go_to_next_frame()
        page.play_button.setText("Play")
        self.play_state = False


    def select_new_leaflet_button_was_pressed(self, page):
        self.draw_leaflet = True
        self.set_buttons_state((True, True, True, False, True), page)
        if self.play_state == True:
            self.play_button_was_pressed()
        #start timer
        page.startTimer()
 
    
    def delete_selected_spline_button_was_pressed(self, page):
        self.update_animation_leaflet_number(page)
        if self.animation.leaflet_nr == 0:
            page.leaflet_pre_plot.removeItem(page.leaflet_pre_graph_ant)
            page.leaflet_pre_graph_ant = None
            page.leaflet_pre_graph_ant = Leaflet_Graph(page.leaflet_pre_plot, page.leaflet_pen, brush=pg.mkBrush("r"), id="pink", tab = page, graph_number =0)
            page.leaflet_pre_plot.addItem(page.leaflet_pre_graph_ant)
            self.initiate_update_leaflet_length_label(0, page)
            self.tabs.tab_3D.mpr_slicer.remove_spline_in_3D(0)

            #remove in 3D
        elif self.animation.leaflet_nr == 1:
            page.leaflet_pre_plot.removeItem(page.leaflet_pre_graph_post)
            page.leaflet_pre_graph_post = None
            page.leaflet_pre_graph_post = Leaflet_Graph(page.leaflet_pre_plot, page.leaflet_pen, brush=pg.mkBrush("g"), id="blue", tab = page, graph_number= 1)
            page.leaflet_pre_plot.addItem(page.leaflet_pre_graph_post)
            self.initiate_update_leaflet_length_label(1, page)
            self.tabs.tab_3D.mpr_slicer.remove_spline_in_3D(1)
        elif self.animation.leaflet_nr == 2:
            page.leaflet_intra_plot.removeItem(page.leaflet_intra_graph_ant)
            page.leaflet_intra_graph_ant = None
            page.leaflet_intra_graph_ant = Leaflet_Graph(page.leaflet_intra_plot, page.leaflet_pen, brush=pg.mkBrush("r"), id="pink", tab = page, graph_number =0)
            page.leaflet_intra_plot.addItem(page.leaflet_intra_graph_ant)
            self.initiate_update_leaflet_length_label(2, page)
            self.tabs.tab_3D.mpr_slicer.remove_spline_in_3D(2)
        elif self.animation.leaflet_nr == 3:
            page.leaflet_intra_plot.removeItem(page.leaflet_intra_graph_post)
            page.leaflet_intra_graph_post = None
            page.leaflet_intra_graph_post = Leaflet_Graph(page.leaflet_intra_plot, page.leaflet_pen, brush=pg.mkBrush("g"), id="blue", tab = page, graph_number= 1)
            page.leaflet_intra_plot.addItem(page.leaflet_intra_graph_post)
            self.initiate_update_leaflet_length_label(3, page)
            self.tabs.tab_3D.mpr_slicer.remove_spline_in_3D(3)
        page.update_scaling()
    
    def select_leaflet_done_button_was_pressed(self, page):
        #stop timer
        
        self.update_animation_leaflet_number(page)
        page.leaflet_number = self.animation.leaflet_nr
        page.stopTimer()
        self.draw_leaflet = False
        self.set_buttons_state((True, True, True, True, False),page)
        if self.animation.leaflet_nr == 0 and self.tabs.tab_3D.mpr_slicer.pre_or_intra == 'pre':
            spline_points_3D_1 = page.leaflet_pre_graph_ant.get_spline_points()
            page.leaflet_pre_graph_ant.add_circles_around_waypoints()
            self.initiate_update_leaflet_length_label(0, page)
            #nothing afterthis function will be complied
            self.tabs.tab_3D.mpr_slicer.show_spline_in_3D(spline_points_3D_1, 1)
            
            
        elif self.animation.leaflet_nr == 1 and self.tabs.tab_3D.mpr_slicer.pre_or_intra == 'pre':
            spline_points_3D_2 = page.leaflet_pre_graph_post.get_spline_points()
            page.leaflet_pre_graph_post.add_circles_around_waypoints()
            self.initiate_update_leaflet_length_label(1, page)
            #nothing afterthis function will be complied
            self.tabs.tab_3D.mpr_slicer.show_spline_in_3D(spline_points_3D_2, 2)
        
        elif self.animation.leaflet_nr == 2 and self.tabs.tab_3D.mpr_slicer.pre_or_intra == 'intra':
            spline_bent_points_3D_1 = page.leaflet_intra_graph_ant.get_spline_points()
            page.leaflet_intra_graph_ant.add_circles_around_waypoints()
            self.initiate_update_leaflet_length_label(2, page)
            #nothing afterthis function will be complied
            self.tabs.tab_3D.mpr_slicer.show_spline_in_3D(spline_bent_points_3D_1, 3)

        elif self.animation.leaflet_nr == 3 and self.tabs.tab_3D.mpr_slicer.pre_or_intra == 'intra':
            spline_bent_points_3D_2 = page.leaflet_intra_graph_post.get_spline_points()
            page.leaflet_intra_graph_post.add_circles_around_waypoints()
            self.initiate_update_leaflet_length_label(3, page)
            #nothing afterthis function will be complied
            self.tabs.tab_3D.mpr_slicer.show_spline_in_3D(spline_bent_points_3D_2, 4)

        if page.bent_mode == True:
            if all(var != 0 for var in (page.leaflet_pre_graph_ant.length_of_leaflet, page.leaflet_pre_graph_post.length_of_leaflet, page.leaflet_intra_graph_ant.length_of_leaflet, page.leaflet_intra_graph_post.length_of_leaflet)):
            
                length_pre_1 = np.round(page.leaflet_pre_graph_ant.length_of_leaflet,2)
                length_pre_2 = np.round(page.leaflet_pre_graph_post.length_of_leaflet,2)
                length_intra_1 = np.round(page.leaflet_intra_graph_ant.length_of_leaflet,2)
                length_intra_2 = np.round(page.leaflet_intra_graph_post.length_of_leaflet,2)
                self.tabs.tab_3D.mpr_slicer.calculate_leaflet_length_differences(length_pre_1, length_pre_2, length_intra_1, length_intra_2)

            

            
    def select_pre_leaflet_button_was_pressed(self, page):
        self.select_pre_leaflet = not self.select_pre_leaflet
        if self.select_pre_leaflet == True:
            page.select_pre_leaflet_button.setText("Select on intra MPR")
        elif self.select_pre_leaflet == False:
            page.select_pre_leaflet_button.setText("Select on pre MPR")


    def initiate_update_leaflet_length_label(self, graph_number, page):
        page.update_leaflet_length_label(graph_number)
            
      

        
    def mouse_clicked(self, ev):
        
        if self.tabs.tab_2D.pre_intra == 'pre':
            vb = self.tabs.tab_2D.leaflet_pre_plot.getPlotItem().getViewBox()
        elif self.tabs.tab_2D.pre_intra == 'intra':
            vb = self.tabs.tab_2D.leaflet_intra_plot.getPlotItem().getViewBox()
        
        
        scene_coords = ev.scenePos()
        self.mouse_pos_in_window= vb.mapSceneToView(scene_coords)

        new_point_x = self.mouse_pos_in_window.x()
        new_point_y = self.mouse_pos_in_window.y()

        if self.draw_leaflet == True:
            self.pass_point_to_spline(new_point_x, new_point_y, self.tabs.tab_2D)

    def mouse_clicked_bent(self, ev):
        if self.tabs.tab_2D_bent.pre_intra == 'pre':
            vb = self.tabs.tab_2D_bent.leaflet_pre_plot.getPlotItem().getViewBox()
        elif self.tabs.tab_2D_bent.pre_intra == 'intra':
            vb = self.tabs.tab_2D_bent.leaflet_intra_plot.getPlotItem().getViewBox()
        scene_coords = ev.scenePos()
        self.mouse_pos_in_window= vb.mapSceneToView(scene_coords)

        new_point_x = self.mouse_pos_in_window.x()
        new_point_y = self.mouse_pos_in_window.y()

        if self.draw_leaflet == True:
            self.pass_point_to_spline(new_point_x, new_point_y, self.tabs.tab_2D_bent)

    def update_animation_leaflet_number(self, page):
        if page.ant_post == 'ant' and page.pre_intra == 'pre':
            self.animation.leaflet_nr = 0
        elif page.ant_post == 'post' and page.pre_intra == 'pre':
            self.animation.leaflet_nr = 1
        elif page.ant_post == 'ant' and page.pre_intra == 'intra':
            self.animation.leaflet_nr = 2
        elif page.ant_post == 'post' and page.pre_intra == 'intra':
            self.animation.leaflet_nr = 3 

    def pass_point_to_spline(self, new_point_x, new_point_y, page):
        self.update_animation_leaflet_number(page)
        
        if self.animation.leaflet_nr == 0:
                polygon_graph = page.leaflet_pre_graph_ant
        elif self.animation.leaflet_nr == 1:
                polygon_graph = page.leaflet_pre_graph_post
        elif self.animation.leaflet_nr == 2:
                polygon_graph = page.leaflet_intra_graph_ant
        elif self.animation.leaflet_nr == 3:
                polygon_graph = page.leaflet_intra_graph_post

        polygon_array = polygon_graph.get_waypoints()
        self.mouse_pos_in_image = [int(new_point_x), int(new_point_y)]
        
        polygon_array[polygon_graph.nr_selected_points_for_polygon, :] = self.mouse_pos_in_image
        polygon_graph.nr_selected_points_for_polygon = polygon_graph.nr_selected_points_for_polygon+1

        self.animation.create_spline(polygon_array, polygon_graph)
        if self.label1_set == False or self.label2_set == False:
            self.set_leaflet_label(page)

        polygon_graph.waypoint_number = polygon_graph.waypoint_number +1

        return


        

    def set_leaflet_label(self, page):
        if self.animation.leaflet_nr == 0:
            page.leaflet_pre_label1.setText('leaflet pre 1')
            self.label1_set = True
        elif self.animation.leaflet_nr == 1:
            page.leaflet_pre_label2.setText('leaflet pre 2')
            self.label2_set = True

        elif self.animation.leaflet_nr == 2:
            page.leaflet_intra_label1.setText('leaflet intra 1')
            self.label_bent1_set = True
        elif self.animation.leaflet_nr == 3:
            page.leaflet_intra_label2.setText('leaflet intra 2')
            self.label_bent2_set = True

    
    
    
    
    
  






