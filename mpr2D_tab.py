import sys

from pydicom import dcmread
from PIL import ImageTk, Image, ImageQt

import numpy as np

import matplotlib.animation as plt_animation
import matplotlib as plt
import matplotlib.pyplot as plt_py
from pydicom.fileset import FileSet
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer,QDateTime, Qt, Slot,QPointF, Signal, QTime
from PySide6.QtGui import QAction, QPainter,QImage, QPixmap, QPen, QIntValidator, QColor, QKeySequence, QPolygon, QPolygonF, QBrush, QMouseEvent, QFont, QCursor
from PySide6.QtWidgets import (QApplication, QLabel,QMainWindow, QPushButton, QWidget, QListWidget,
                               QLineEdit,QFileDialog,QVBoxLayout, QHBoxLayout,QDialogButtonBox,
                               QGridLayout, QCheckBox, QMessageBox, QSizePolicy, QSlider, QGraphicsView, 
                               QGraphicsScene, QStyle,QSpacerItem, QRadioButton, QTabWidget, QGroupBox)
from random import randint, choice
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import axes3d
import matplotlib.patches as mpatches
import pyqtgraph as pg
from scipy.interpolate import CubicSpline, UnivariateSpline, LSQUnivariateSpline
import keyboard
from matplotlib import cm
from matplotlib import colormaps

from leaflet_graph import Leaflet_Graph
from mpr_slicer import MPR_Slicer 




class MPR_2D_Tab(QWidget):
    def __init__(self, mpr_slicer, bent_mode, excel_writer):
        super().__init__()

        self.excel_writer = excel_writer
        
        self.leaflet_pen = QPen(Qt.red, 3)
        self.orange_pen = pg.mkPen(color=(255, 165, 0), width=2)
        self.ant_post = 'ant'
        self.pre_intra = 'pre'


       
        self.mpr_slicer = mpr_slicer #MPR_Slicer() 
        self.bent_mode = bent_mode
        if self.bent_mode == False:
            self.mpr_slicer.tab_2D = self
        elif self.bent_mode == True:
            self.mpr_slicer.tab_2D_bent = self

        

        if self.bent_mode == False: 
            text_pre = 'MPR through clip - pre op'
            text_intra = 'MPR through clip - intra op'
        elif self.bent_mode == True: 
            text_pre = 'Test bent leaflet - pre op'
            text_intra = 'Test bent leaflet - intra op'


        #timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateTimer)
        self.currentTime = QTime(0, 0, 0)
        self.startTime = QTime(0, 0, 0)
        self.running = False

        self.leaflet_number =0

        self.selection_time_ant_pre = 0
        self.selection_time_post_pre = 0
        self.selection_time_ant_intra = 0
        self.selection_time_post_intra = 0

        self.intersection_1_pre = None
        self.intersection_2_pre = None
        self.intersection_1_intra = None
        self.intersection_2_intra = None
        self.intersection_1_bent_pre = None
        self.intersection_2_bent_pre = None
        self.intersection_1_bent_intra = None
        self.intersection_2_bent_intra = None
        self.intersection_1_neighbor = None
        self.intersection_2_neighbor = None

        self.all_intersections = [self.intersection_1_pre, self.intersection_2_pre, self.intersection_1_intra, 
                                  self.intersection_2_intra, self.intersection_1_bent_pre, self.intersection_2_bent_pre,
                                  self.intersection_1_bent_intra, self.intersection_2_bent_intra, 
                                  self.intersection_1_neighbor, self.intersection_2_neighbor]


        self.leaflet_pre_plot = pg.PlotWidget()
        self.leaflet_pre_graph_ant = Leaflet_Graph(self.leaflet_pre_plot, self.leaflet_pen, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =0)
        self.leaflet_pre_graph_post = Leaflet_Graph(self.leaflet_pre_plot, self.leaflet_pen, brush=pg.mkBrush("g"), id="blue", tab = self, graph_number= 1)
        self.leaflet_pre_label0 = pg.TextItem(text= text_pre, anchor=(0, 3), color= (255, 255, 255))
        self.leaflet_pre_label1 = pg.TextItem(text='', anchor=(0, 2), color= (200, 0, 0))
        self.leaflet_pre_label2 = pg.TextItem(text='', anchor=(0, 1), color= (0, 200, 0))
        self.mpr_pre_image = pg.ImageItem()

        self.leaflet_intra_plot = pg.PlotWidget()
        self.leaflet_intra_graph_ant = Leaflet_Graph(self.leaflet_intra_plot, self.leaflet_pen, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =0)
        self.leaflet_intra_graph_post = Leaflet_Graph(self.leaflet_intra_plot, self.leaflet_pen, brush=pg.mkBrush("g"), id="blue", tab = self, graph_number= 1)
        self.leaflet_intra_label0 = pg.TextItem(text=text_intra, anchor=(0, 3), color= (255, 255, 255))
        self.leaflet_intra_label1 = pg.TextItem(text='', anchor=(0, 2), color= (200, 0, 0))
        self.leaflet_intra_label2 = pg.TextItem(text='', anchor=(0, 1), color= (0, 200, 0))
        self.mpr_intra_image = pg.ImageItem()


        self.ant_spline_radio_button = QRadioButton("select anterior spline")
        self.ant_spline_radio_button.setChecked(True)
        self.ant_spline_radio_button.toggled.connect(self.ant_spline_radio_button_selected)
        self.post_spline_radio_button = QRadioButton("select posterior spline")
        self.post_spline_radio_button.setChecked(False)
        self.post_spline_radio_button.toggled.connect(self.post_spline_radio_button_selected)

        self.pre_spline_radio_button = QRadioButton("select pre procedural spline")
        self.pre_spline_radio_button.setChecked(True)
        self.pre_spline_radio_button.toggled.connect(self.pre_spline_radio_button_selected)
        self.intra_spline_radio_button = QRadioButton("select intra procedural spline")
        self.intra_spline_radio_button.setChecked(False)
        self.intra_spline_radio_button.toggled.connect(self.intra_spline_radio_button_selected)


        FirstExclusiveGroup = QGroupBox("Choose which leaflet you want to select")
        SecondExclusiveGroup = QGroupBox("Choose which leaflet you want to select")

        vbox_1= QVBoxLayout()
        vbox_2= QVBoxLayout()
        FirstExclusiveGroup.setLayout(vbox_1)
        SecondExclusiveGroup.setLayout(vbox_2)
        
        vbox_1.addWidget(self.ant_spline_radio_button)
        vbox_1.addWidget(self.post_spline_radio_button) 
        vbox_2.addWidget(self.pre_spline_radio_button)
        vbox_2.addWidget(self.intra_spline_radio_button)


        colorMap = pg.colormap.get('gray', source = 'matplotlib')
        self.mpr_pre_image.setColorMap(colorMap)
        self.mpr_intra_image.setColorMap(colorMap)

        self.label_layout = QVBoxLayout()
        self.leaflet_introduction_label = QLabel(text='Start with the pre op anterior leaflet\nand then select\nthe pre op posterior,\nintra op anterior and\nlastly the intra op posterior leaflet.')
        self.leaflet_length0_label = QLabel(text='Leaflet lengths:')
        self.leaflet_pre_length1_label = QLabel(text='')
        self.leaflet_pre_length2_label = QLabel(text='')
        self.leaflet_intra_length1_label = QLabel(text='')
        self.leaflet_intra_length2_label = QLabel(text='')
        self.timerLabel = QLabel("Timer: 00:00:00")
        self.label_layout.addWidget(self.leaflet_length0_label)
        self.label_layout.addWidget(self.leaflet_pre_length1_label)
        self.label_layout.addWidget(self.leaflet_pre_length2_label)
        self.label_layout.addWidget(self.leaflet_intra_length1_label)
        self.label_layout.addWidget(self.leaflet_intra_length2_label)
        self.label_layout.addWidget(self.timerLabel)

        if self.bent_mode == False: 
            self.mpr_pre_image.setImage(self.mpr_slicer.get_pre_mpr())
            self.mpr_pre_image.setLevels([0,255])
            self.mpr_intra_image.setImage(self.mpr_slicer.get_intra_mpr())
            self.mpr_intra_image.setLevels([0,255])

        elif self.bent_mode == True: 
            self.mpr_pre_image.setImage(self.mpr_slicer.get_bent_pre_mpr())
            self.mpr_pre_image.setLevels([0,255])
            self.mpr_intra_image.setImage(self.mpr_slicer.get_bent_intra_mpr())
            self.mpr_intra_image.setLevels([0,255])
        
    
        self.leaflet_pre_plot.addItem(self.mpr_pre_image)
        self.leaflet_pre_plot.addItem(self.leaflet_pre_graph_ant)
        self.leaflet_pre_plot.addItem(self.leaflet_pre_graph_post)
        self.leaflet_pre_plot.addItem(self.leaflet_pre_label0)
        self.leaflet_pre_plot.addItem(self.leaflet_pre_label1)
        self.leaflet_pre_plot.addItem(self.leaflet_pre_label2)

        self.leaflet_intra_plot.addItem(self.mpr_intra_image)
        self.leaflet_intra_plot.addItem(self.leaflet_intra_graph_ant)
        self.leaflet_intra_plot.addItem(self.leaflet_intra_graph_post)
        self.leaflet_intra_plot.addItem(self.leaflet_intra_label0)
        self.leaflet_intra_plot.addItem(self.leaflet_intra_label1)
        self.leaflet_intra_plot.addItem(self.leaflet_intra_label2)

        self.leaflet_pre_plot.getPlotItem().hideAxis("left")
        self.leaflet_pre_plot.getPlotItem().hideAxis("bottom")

        self.leaflet_intra_plot.getPlotItem().hideAxis("left")
        self.leaflet_intra_plot.getPlotItem().hideAxis("bottom")
        
        y0 = 0
        y1 = self.mpr_slicer.mpr_height
        x0= 0
        x1 = self.mpr_slicer.mpr_width
        self.leaflet_pre_plot.getPlotItem().setYRange(y0, y1)
        self.leaflet_pre_plot.getPlotItem().setXRange(x0, x1)
        self.ratio = (x1-x0)/ (y1-y0)
        self.fixed_image_size =1000
        self.leaflet_pre_plot.setFixedSize(self.fixed_image_size*self.ratio, self.fixed_image_size)
        self.leaflet_intra_plot.setFixedSize(self.fixed_image_size*self.ratio, self.fixed_image_size)

        
        self.leaflet_pre_plot.getPlotItem().setMouseEnabled(x=False, y=False)
        self.leaflet_intra_plot.getPlotItem().setMouseEnabled(x=False, y=False)
        
        button_layout = QHBoxLayout()
        label_2D_layout = QHBoxLayout()
        label_2D_layout.addWidget(self.leaflet_pre_plot)
        label_2D_layout.addWidget(self.leaflet_intra_plot)
        
        self.play_button = QPushButton("Play")
        self.helper_button = QPushButton("Stop")
        self.last_button = QPushButton("Last")
        self.next_button = QPushButton("Next")
        self.select_leaflet_button =QPushButton("Select new leaflet")
        self.select_leaflet_done_button =QPushButton("Selection Done")
        self.select_pre_leaflet_button =QPushButton("Select on intra mpr")
        self.delete_selected_spline_button =QPushButton("Delete selected spline")

        self.play_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.last_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.next_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.select_leaflet_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.select_leaflet_done_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.select_pre_leaflet_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.delete_selected_spline_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.leaflet_pre_plot.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.leaflet_intra_plot.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        save_button_layout = QHBoxLayout()
        self.save_with_complete_model_button = QPushButton("Save lengths with complete model")
        self.save_with_complete_model_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.save_with_complete_model_button.clicked.connect(self.save_with_complete_model_button_was_pressed)
        self.save_with_partial_model_button = QPushButton("Save lengths with partial model")
        self.save_with_partial_model_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.save_with_partial_model_button.clicked.connect(self.save_with_partial_model_button_was_pressed)
        self.save_without_model_button = QPushButton("Save lengths without model")
        self.save_without_model_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.save_without_model_button.clicked.connect(self.save_without_model_button_was_pressed)
        self.save_with_neighbor_slice_button = QPushButton("Save lengths with neighbor slice")
        self.save_with_neighbor_slice_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.save_with_neighbor_slice_button.clicked.connect(self.save_with_neighbor_slice_button_was_pressed)

        save_button_layout.addWidget(self.save_with_complete_model_button)
        save_button_layout.addWidget(self.save_with_partial_model_button)
        save_button_layout.addWidget(self.save_without_model_button)
        save_button_layout.addWidget(self.save_with_neighbor_slice_button)

        save_bent_button_layout = QHBoxLayout()
        self.save_bent_lengths_button = QPushButton("Save bent lengths")
        self.save_bent_lengths_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.save_bent_lengths_button.clicked.connect(self.save_bent_lengths_button_was_pressed)
        save_bent_button_layout.addWidget(self.save_bent_lengths_button)

        #add the buttons to the gui
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.last_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.select_leaflet_button)
        button_layout.addWidget(self.select_leaflet_done_button)
        button_layout.addWidget(self.delete_selected_spline_button)

    
        window_layout = QGridLayout()
        window_layout.addLayout(self.label_layout,0,0)
        window_layout.addLayout(label_2D_layout, 0,1)
        window_layout.addLayout(button_layout, 1,1)
        if self.bent_mode == False:
            window_layout.addLayout(save_button_layout,2,1)
        elif self.bent_mode == True:
            window_layout.addLayout(save_bent_button_layout,2,1)
        window_layout.addWidget(FirstExclusiveGroup, 3,0)
        window_layout.addWidget(SecondExclusiveGroup, 4,0)
        
        
        self.setLayout(window_layout)

    
    def ant_spline_radio_button_selected(self):
        self.ant_post = 'ant'
    def post_spline_radio_button_selected(self):
        self.ant_post = 'post'
    def pre_spline_radio_button_selected(self):
        self.pre_intra = 'pre'
    def intra_spline_radio_button_selected(self):
        self.pre_intra = 'intra'

    def update_mpr(self):
        if self.bent_mode == False:
            try: 
                self.remove_annlus(self.leaflet_pre_plot, pre =True, mode = 'normal')
            except:
                pass
            try:
                self.remove_annlus(self.leaflet_intra_plot, pre =False, mode = 'normal')
            except:
                pass
            self.mpr_pre_image.setImage(self.mpr_slicer.get_pre_mpr()) 
            self.mpr_intra_image.setImage(self.mpr_slicer.get_intra_mpr()) 
            self.set_annulus(self.mpr_slicer.get_pre_annulus(), self.leaflet_pre_plot, pre =True, mode = 'normal')
            self.set_annulus(self.mpr_slicer.get_intra_annulus(), self.leaflet_intra_plot, pre =False, mode = 'normal')
        elif self.bent_mode == True:
            try: 
                self.remove_annlus(self.leaflet_pre_plot, pre =True, mode = 'bent')
            except:
                pass
            try:
                self.remove_annlus(self.leaflet_intra_plot, pre =False, mode = 'bent')
            except:
                pass
            self.mpr_pre_image.setImage(self.mpr_slicer.get_bent_pre_mpr()) 
            self.mpr_intra_image.setImage(self.mpr_slicer.get_bent_intra_mpr()) 
            self.set_annulus(self.mpr_slicer.get_bent_pre_annulus(), self.leaflet_pre_plot, pre =True, mode = 'bent')
            self.set_annulus(self.mpr_slicer.get_bent_intra_annulus(), self.leaflet_intra_plot,  pre =False, mode = 'bent')
            
        self.mpr_pre_image.setLevels([0,255])
        self.mpr_intra_image.setLevels([0,255])


    def insert_arbitrary_image(self, arbitrary_image):
        self.mpr_pre_image.setImage(arbitrary_image)
        x0 = 0
        y0 = 0
        x1 = arbitrary_image.shape[1]
        y1 = arbitrary_image.shape[0]
        self.leaflet_pre_plot.getPlotItem().setYRange(y0, y1)
        self.leaflet_pre_plot.getPlotItem().setXRange(x0, x1)
        ratio = (x1-x0)/ (y1-y0)

    def set_annulus(self, annulus_points, plot_item, pre, mode):
        if annulus_points is not None:
            if not (annulus_points[0,:] == 0).all():
                x_1 = annulus_points[0,0]
                y_1 = annulus_points[0,1]
                
                if pre == True and mode == 'normal':
                    self.intersection_1_pre = plot_item.plot([x_1], [y_1], symbol='x', symbolPen = self.orange_pen, symbolBrush=(255, 165, 0), symbolSize=12)

                if pre == False and mode == 'normal':
                    self.intersection_1_intra = plot_item.plot([x_1], [y_1], symbol='x', symbolPen = self.orange_pen, symbolBrush=(255, 165, 0), symbolSize=12)

                if pre == True and mode == 'bent':
                    self.intersection_1_bent_pre = plot_item.plot([x_1], [y_1], symbol='x', symbolPen = self.orange_pen, symbolBrush=(255, 165, 0), symbolSize=12)

                if pre == False and mode == 'bent':
                    self.intersection_1_bent_intra = plot_item.plot([x_1], [y_1], symbol='x', symbolPen = self.orange_pen, symbolBrush=(255, 165, 0), symbolSize=12)
                    
            if not (annulus_points[1,:] == 0).all():
                x_2 = annulus_points[1,0]
                y_2 = annulus_points[1,1]

                if pre == True and mode == 'normal':
                    self.intersection_2_pre = plot_item.plot([x_2], [y_2], symbol='x', symbolPen = self.orange_pen, symbolBrush=(255, 165, 0), symbolSize=12)

                if pre == False and mode == 'normal':
                    self.intersection_2_intra = plot_item.plot([x_2], [y_2], symbol='x', symbolPen = self.orange_pen, symbolBrush=(255, 165, 0), symbolSize=12)

                if pre == True and mode == 'bent':
                    self.intersection_2_bent_pre = plot_item.plot([x_2], [y_2], symbol='x', symbolPen = self.orange_pen, symbolBrush=(255, 165, 0), symbolSize=12)

                if pre == False and mode == 'bent':
                    self.intersection_2_bent_intra = plot_item.plot([x_2], [y_2], symbol='x', symbolPen = self.orange_pen, symbolBrush=(255, 165, 0), symbolSize=12)


    def remove_annlus(self, plot_item, pre, mode):
        if pre == True and mode == 'normal':
           plot_item.removeItem(self.intersection_1_pre)
           plot_item.removeItem(self.intersection_2_pre)

        if pre == False and mode == 'normal':
           plot_item.removeItem(self.intersection_1_intra)
           plot_item.removeItem(self.intersection_2_intra)

        if pre == True and mode == 'bent':
           plot_item.removeItem(self.intersection_1_bent_pre)
           plot_item.removeItem(self.intersection_2_bent_pre)

        if pre == False and mode == 'bent':
           plot_item.removeItem(self.intersection_1_bent_intra)
           plot_item.removeItem(self.intersection_2_bent_intra)

        

    def import_arbitrary_image(self):
        path = #
        arbitrary_image = Image.open(path)
        arbitrary_image_array = np.array(arbitrary_image)
        return arbitrary_image_array
    
    def set_scaling_factor(self,x_scale, y_scale):
        self.scaling_factor = [x_scale, y_scale]

    def update_scaling(self):
        self.leaflet_pre_graph_ant.leaflet_scaling = self.scaling_factor
        self.leaflet_pre_graph_post.leaflet_scaling = self.scaling_factor
        self.leaflet_intra_graph_ant.leaflet_scaling = self.scaling_factor
        self.leaflet_intra_graph_post.leaflet_scaling = self.scaling_factor

    def startTimer(self):
        if not self.running:
            self.startTime = QTime.currentTime()
            self.currentTime = QTime(0, 0, 0)
            self.timer.start(1000)
            self.running = True
            self.updateTimer()

    def stopTimer(self):
        if self.running:
            self.timer.stop()
            self.running = False
        self.save_time()

    def save_time(self):
        if self.leaflet_number ==0:
            self.selection_time_ant_pre = self.currentTime
        elif self.leaflet_number ==1:
            self.selection_time_post_pre = self.currentTime
        elif self.leaflet_number ==2:
            self.selection_time_ant_intra = self.currentTime
        elif self.leaflet_number ==3:
            self.selection_time_post_intra = self.currentTime
        
            

    def updateTimer(self):
        elapsed = self.startTime.secsTo(QTime.currentTime())
        self.currentTime = self.currentTime.addSecs(1)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.timerLabel.setText(f"Timer: {hours:02}:{minutes:02}:{seconds:02}")



    def update_leaflet_length_label(self, graph_number):
        text_len_1 = 'length leaflet pre 1: ' +str(round(self.leaflet_pre_graph_ant.length_of_leaflet,2))+'mm'
        self.leaflet_pre_length1_label.setText(text_len_1)

    
        text_len_2 = 'length leaflet pre 2: ' +str(round(self.leaflet_pre_graph_post.length_of_leaflet,2))+'mm'
        self.leaflet_pre_length2_label.setText(text_len_2)

    
        text_len_3 = 'length leaflet intra 1: ' +str(round(self.leaflet_intra_graph_ant.length_of_leaflet,2))+'mm'
        self.leaflet_intra_length1_label.setText(text_len_3)

    
        text_len_4 = 'length leaflet intra 2: ' +str(round(self.leaflet_intra_graph_post.length_of_leaflet,2))+'mm'
        self.leaflet_intra_length2_label.setText(text_len_4)


    def save_with_complete_model_button_was_pressed(self):
        self.length_complete_ant_total = self.mpr_slicer.total_leaflet_length_ant
        self.length_complete_ant_visible = self.leaflet_intra_graph_ant.length_of_leaflet

        self.length_complete_post_total = self.mpr_slicer.total_leaflet_length_post 
        self.length_complete_post_visible = self.leaflet_intra_graph_post.length_of_leaflet

        self.excel_writer.add_value('pre', 'length_complete_ant', self.length_complete_ant_total)
        self.excel_writer.add_value('pre', 'length_complete_post', self.length_complete_post_total)
        self.excel_writer.add_value('intra', 'length_complete_ant', self.length_complete_ant_visible)
        self.excel_writer.add_value('intra', 'length_complete_post', self.length_complete_post_visible)

        self.excel_writer.add_value('pre', 'selection_time_ant_complete', self.selection_time_ant_pre)
        self.excel_writer.add_value('intra', 'selection_time_ant_complete', self.selection_time_ant_intra)
        self.excel_writer.add_value('pre', 'selection_time_post_complete', self.selection_time_post_pre)
        self.excel_writer.add_value('intra', 'selection_time_post_complete', self.selection_time_post_intra)

        self.excel_writer.add_value('pre', 'number_waypoints_ant_complete', self.leaflet_pre_graph_ant.waypoint_number)
        self.excel_writer.add_value('intra', 'number_waypoints_ant_complete', self.leaflet_intra_graph_ant.waypoint_number)
        self.excel_writer.add_value('pre', 'number_waypoints_post_complete', self.leaflet_pre_graph_post.waypoint_number)
        self.excel_writer.add_value('intra', 'number_waypoints_post_complete', self.leaflet_intra_graph_post.waypoint_number)

        print('Leaflet lengths for complete model were saved.')
        
    def save_with_partial_model_button_was_pressed(self):
        self.length_partial_ant_total = self.mpr_slicer.total_leaflet_length_ant
        self.length_partial_ant_visible = self.leaflet_intra_graph_ant.length_of_leaflet

        self.length_partial_post_total = self.mpr_slicer.total_leaflet_length_post 
        self.length_partial_post_visible = self.leaflet_intra_graph_post.length_of_leaflet

        self.excel_writer.add_value('pre', 'length_partial_ant', self.length_partial_ant_total)
        self.excel_writer.add_value('pre', 'length_partial_post', self.length_partial_post_total)
        self.excel_writer.add_value('intra', 'length_partial_ant', self.length_partial_ant_visible)
        self.excel_writer.add_value('intra', 'length_partial_post', self.length_partial_post_visible)

        self.excel_writer.add_value('pre', 'selection_time_ant_partial', self.selection_time_ant_pre)
        self.excel_writer.add_value('intra', 'selection_time_ant_partial', self.selection_time_ant_intra)
        self.excel_writer.add_value('pre', 'selection_time_post_partial', self.selection_time_post_pre)
        self.excel_writer.add_value('intra', 'selection_time_post_partial', self.selection_time_post_intra)

        self.excel_writer.add_value('pre', 'number_waypoints_ant_partial', self.leaflet_pre_graph_ant.waypoint_number)
        self.excel_writer.add_value('intra', 'number_waypoints_ant_partial', self.leaflet_intra_graph_ant.waypoint_number)
        self.excel_writer.add_value('pre', 'number_waypoints_post_partial', self.leaflet_pre_graph_post.waypoint_number)
        self.excel_writer.add_value('intra', 'number_waypoints_post_partial', self.leaflet_intra_graph_post.waypoint_number)

        print('Leaflet lengths for partial model were saved.')

    def save_without_model_button_was_pressed(self):
        self.length_without_ant_total = self.leaflet_pre_graph_ant.length_of_leaflet 
        self.length_without_ant_visible = self.leaflet_intra_graph_ant.length_of_leaflet

        self.length_without_post_total = self.leaflet_pre_graph_post.length_of_leaflet 
        self.length_without_post_visible = self.leaflet_intra_graph_post.length_of_leaflet

        self.excel_writer.add_value('pre', 'length_without_ant', self.length_without_ant_total)
        self.excel_writer.add_value('pre', 'length_without_post', self.length_without_post_total)
        self.excel_writer.add_value('intra', 'length_without_ant', self.length_without_ant_visible)
        self.excel_writer.add_value('intra', 'length_without_post', self.length_without_post_visible)

        self.excel_writer.add_value('pre', 'selection_time_ant_without', self.selection_time_ant_pre)
        self.excel_writer.add_value('intra', 'selection_time_ant_without', self.selection_time_ant_intra)
        self.excel_writer.add_value('pre', 'selection_time_post_without', self.selection_time_post_pre)
        self.excel_writer.add_value('intra', 'selection_time_post_without', self.selection_time_post_intra)

        self.excel_writer.add_value('pre', 'number_waypoints_ant_without', self.leaflet_pre_graph_ant.waypoint_number)
        self.excel_writer.add_value('intra', 'number_waypoints_ant_without', self.leaflet_intra_graph_ant.waypoint_number)
        self.excel_writer.add_value('pre', 'number_waypoints_post_without', self.leaflet_pre_graph_post.waypoint_number)
        self.excel_writer.add_value('intra', 'number_waypoints_post_without', self.leaflet_intra_graph_post.waypoint_number)

        print('Leaflet lengths for without model were saved.')

    def save_with_neighbor_slice_button_was_pressed(self):
        self.length_neighbor_ant_total = self.leaflet_pre_graph_ant.length_of_leaflet 
        self.length_neighbor_ant_visible = self.leaflet_intra_graph_ant.length_of_leaflet

        self.length_neighbor_post_total = self.leaflet_pre_graph_post.length_of_leaflet 
        self.length_neighbor_post_visible = self.leaflet_intra_graph_post.length_of_leaflet

        self.excel_writer.add_value('pre', 'length_neighbor_ant', self.length_neighbor_ant_total)
        self.excel_writer.add_value('pre', 'length_neighbor_post', self.length_neighbor_post_total)
        self.excel_writer.add_value('intra', 'length_neighbor_ant', self.length_neighbor_ant_visible)
        self.excel_writer.add_value('intra', 'length_neighbor_post', self.length_neighbor_post_visible)

        self.excel_writer.add_value('pre', 'selection_time_ant_neighbor', self.selection_time_ant_pre)
        self.excel_writer.add_value('intra', 'selection_time_ant_neighbor', self.selection_time_ant_intra)
        self.excel_writer.add_value('pre', 'selection_time_post_neighbor', self.selection_time_post_pre)
        self.excel_writer.add_value('intra', 'selection_time_post_neighbor', self.selection_time_post_intra)

        self.excel_writer.add_value('pre', 'number_waypoints_ant_neighbor', self.leaflet_pre_graph_ant.waypoint_number)
        self.excel_writer.add_value('intra', 'number_waypoints_ant_neighbor', self.leaflet_intra_graph_ant.waypoint_number)
        self.excel_writer.add_value('pre', 'number_waypoints_post_neighbor', self.leaflet_pre_graph_post.waypoint_number)
        self.excel_writer.add_value('intra', 'number_waypoints_post_neighbor', self.leaflet_intra_graph_post.waypoint_number)

        print('Leaflet lengths for neighbor slice were saved.')

    def save_bent_lengths_button_was_pressed(self):
        self.length_bent_ant_total = self.leaflet_pre_graph_ant.length_of_leaflet 
        self.length_bent_ant_visible = self.leaflet_intra_graph_ant.length_of_leaflet

        self.length_bent_post_total = self.leaflet_pre_graph_post.length_of_leaflet 
        self.length_bent_post_visible = self.leaflet_intra_graph_post.length_of_leaflet

        self.excel_writer.add_value('pre', 'length_bent_ant', self.length_bent_ant_total)
        self.excel_writer.add_value('pre', 'length_bent_post', self.length_bent_post_total)
        self.excel_writer.add_value('intra', 'length_bent_ant', self.length_bent_ant_visible)
        self.excel_writer.add_value('intra', 'length_bent_post', self.length_bent_post_visible)

        self.excel_writer.add_value('pre', 'selection_time_ant_bent', self.selection_time_ant_pre)
        self.excel_writer.add_value('intra', 'selection_time_ant_bent', self.selection_time_ant_intra)
        self.excel_writer.add_value('pre', 'selection_time_post_bent', self.selection_time_post_pre)
        self.excel_writer.add_value('intra', 'selection_time_post_bent', self.selection_time_post_intra)

        self.excel_writer.add_value('pre', 'number_waypoints_ant_bent', self.leaflet_pre_graph_ant.waypoint_number)
        self.excel_writer.add_value('intra', 'number_waypoints_ant_bent', self.leaflet_intra_graph_ant.waypoint_number)
        self.excel_writer.add_value('pre', 'number_waypoints_post_bent', self.leaflet_pre_graph_post.waypoint_number)
        self.excel_writer.add_value('intra', 'number_waypoints_post_bent', self.leaflet_intra_graph_post.waypoint_number)

        print('Leaflet lengths for bent slice were saved.')










































        