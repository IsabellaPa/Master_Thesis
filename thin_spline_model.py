import sys

import numpy as np
import random
import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import axes3d
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QPainter,QImage, QPixmap, QPen, QIntValidator, QColor, QKeySequence, QPolygon, QPolygonF, QBrush, QMouseEvent, QFont, QCursor
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout,
                               QHeaderView, QLabel, QMainWindow, QSlider,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget, QGridLayout, QRadioButton, QLineEdit, QFormLayout, QPushButton, QSizePolicy,QDialog, QFileDialog)

from mpr_slicer import MPR_Slicer
from model3D import Model_3D
from viewer3D import VtkDisplay
import pathlib
import os
from leaflet_graph import Leaflet_Graph
from scipy.interpolate import CubicSpline, UnivariateSpline, LSQUnivariateSpline
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from vedo import *


class Thin_spline_model(QWidget):
    def __init__(self, mpr_slicer, excel_writer, dataloader):
        super().__init__()
        self.mpr_slicer = mpr_slicer
        self.excel_writer = excel_writer 
        self.dataloader = dataloader
        self.animation = None

        self.mode_anterior_posterior = 'ant'
        self.mpr_array_parallel_pre = None
        self.mpr_array_parallel_intra = None
        self.mpr_array_rotated_pre = None
        self.mpr_array_rotated_intra = None
        self.current_mpr_array = self.mpr_array_parallel_intra

        self.mode_parallel = True
        self.pre_mpr = False

        self.current_mpr = 0

        

        self.anterior_radio_button = QRadioButton('anterior')
        self.anterior_radio_button.clicked.connect(self.anterior_radio_button_was_pressed)
        self.anterior_radio_button.setChecked(True)
        self.posterior_radio_button = QRadioButton('posterior')
        self.posterior_radio_button.clicked.connect(self.posterior_radio_button_was_pressed)
        self.posterior_radio_button.setChecked(False)

        self.next_button = QPushButton('Next')
        self.next_button.clicked.connect(self.next_button_was_pressed)
        self.back_button = QPushButton('Back')
        self.back_button.clicked.connect(self.back_button_was_pressed)
        self.back_button.setEnabled(False)

        self.delete_selected_spline_button = QPushButton('Delete selected spline')
        self.delete_selected_spline_button.clicked.connect(self.delete_selected_spline_button_was_pressed)
        

        self.get_mprs_parallel_button = QPushButton('Get MPRs parallel')
        self.get_mprs_parallel_button.clicked.connect(self.get_mprs_parallel_button_was_pressed)
        self.get_mprs_rotated_button = QPushButton('Get MPRs rotated')
        self.get_mprs_rotated_button.clicked.connect(self.get_mprs_rotated_button_was_pressed)

        self.change_pre_intra_button = QPushButton('Change to pre')
        self.change_pre_intra_button.clicked.connect(self.change_pre_intra_button_was_pressed)
        self.change_mode_button = QPushButton('Change to rotated')
        self.change_mode_button.clicked.connect(self.change_mode_button_was_pressed)
        self.create_model_parallel_pre_button = QPushButton('Create parallel pre model')
        self.create_model_parallel_pre_button.clicked.connect(self.create_model_parallel_pre_button_was_pressed)
        self.create_model_parallel_intra_button = QPushButton('Create parallel intra model')
        self.create_model_parallel_intra_button.clicked.connect(self.create_model_parallel_intra_button_was_pressed)
        self.create_model_rotated_pre_button = QPushButton('Create rotated pre model')
        self.create_model_rotated_pre_button.clicked.connect(self.create_model_rotated_pre_button_was_pressed)
        self.create_model_rotated_intra_button = QPushButton('Create rotated intra model')
        self.create_model_rotated_intra_button.clicked.connect(self.create_model_rotated_intra_button_was_pressed)
        self.save_model_parallel_pre_button = QPushButton('Save parallel pre model')
        self.save_model_parallel_pre_button.clicked.connect(self.save_model_parallel_pre_button_was_pressed)
        self.save_model_parallel_intra_button = QPushButton('Save parallel intra model')
        self.save_model_parallel_intra_button.clicked.connect(self.save_model_parallel_intra_button_was_pressed)
        self.save_model_rotated_pre_button = QPushButton('Save rotated pre model')
        self.save_model_rotated_pre_button.clicked.connect(self.save_model_rotated_pre_button_was_pressed)
        self.save_model_rotated_intra_button = QPushButton('Save rotated intra model')
        self.save_model_rotated_intra_button.clicked.connect(self.save_model_rotated_intra_button_was_pressed)

        self.button_layout= QGridLayout()
        self.button_layout.addWidget(self.anterior_radio_button, 0,0)
        self.button_layout.addWidget(self.posterior_radio_button, 0,1)
        self.button_layout.addWidget(self.next_button, 1,1)
        self.button_layout.addWidget(self.back_button, 1,0)
        self.button_layout.addWidget(self.delete_selected_spline_button, 1,2)
        self.button_layout.addWidget(self.get_mprs_parallel_button, 2,0)
        self.button_layout.addWidget(self.get_mprs_rotated_button, 2,1)
        self.button_layout.addWidget(self.change_pre_intra_button, 3,0)
        self.button_layout.addWidget(self.change_mode_button, 3,1)
        self.button_layout.addWidget(self.create_model_parallel_pre_button, 4,0)
        self.button_layout.addWidget(self.create_model_rotated_pre_button, 4,1)
        self.button_layout.addWidget(self.create_model_parallel_intra_button, 4,2)
        self.button_layout.addWidget(self.create_model_rotated_intra_button, 4,3)
        self.button_layout.addWidget(self.save_model_parallel_pre_button, 5,0)
        self.button_layout.addWidget(self.save_model_rotated_pre_button, 5,1)
        self.button_layout.addWidget(self.save_model_parallel_intra_button, 5,2)
        self.button_layout.addWidget(self.save_model_rotated_intra_button, 5,3)



        self.mpr_plot = pg.PlotWidget()
        self.mpr_image = pg.ImageItem()
        colorMap = pg.colormap.get('gray', source = 'matplotlib')
        self.mpr_image.setColorMap(colorMap)
        self.mpr_image.setLevels([0, 255])
        self.image_height = self.dataloader.dim_size_intra[2]
        self.image_width = self.dataloader.dim_size_intra[1]

        self.mpr_plot.addItem(self.mpr_image)
        self.mpr_plot.getPlotItem().hideAxis("left")
        self.mpr_plot.getPlotItem().hideAxis("bottom")
        self.mpr_plot.getPlotItem().setMouseEnabled(x=False, y=False)

        y0 = 0
        y1 = self.mpr_slicer.mpr_height
        x0= 0
        x1 = self.mpr_slicer.mpr_width
        self.mpr_plot.getPlotItem().setYRange(y0, y1)
        self.mpr_plot.getPlotItem().setXRange(x0, x1)
        self.ratio = (x1-x0)/ (y1-y0)
        self.fixed_image_size =1000
        self.mpr_plot.setFixedSize(self.fixed_image_size*self.ratio, self.fixed_image_size)

        self.orange_pen = pg.mkPen(color=(255, 165, 0), width=2)


        self.plot_layout = QHBoxLayout()
        self.plot_layout.addWidget(self.mpr_plot)


        self.leaflet_pen_1 = QPen(Qt.red, 3)
        self.leaflet_pen_2 = QPen(Qt.red, 3)

        self.leaflet_anterior_parallel_pre_1= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =0)
        self.leaflet_anterior_parallel_intra_1 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =1)
        self.leaflet_anterior_rotated_pre_1= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =2)
        self.leaflet_anterior_rotated_intra_1 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =3)
        self.leaflet_posterior_parallel_pre_1= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =4)
        self.leaflet_posterior_parallel_intra_1 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =5)
        self.leaflet_posterior_rotated_pre_1= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =6)
        self.leaflet_posterior_rotated_intra_1 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =7)
        
        self.leaflet_anterior_parallel_pre_2= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =8)
        self.leaflet_anterior_parallel_intra_2 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =9)
        self.leaflet_anterior_rotated_pre_2= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =10)
        self.leaflet_anterior_rotated_intra_2 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =11)
        self.leaflet_posterior_parallel_pre_2= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =12)
        self.leaflet_posterior_parallel_intra_2 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =13)
        self.leaflet_posterior_rotated_pre_2= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =14)
        self.leaflet_posterior_rotated_intra_2 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =15)
        
        self.leaflet_anterior_parallel_pre_3= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =16)
        self.leaflet_anterior_parallel_intra_3 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =17)
        self.leaflet_anterior_rotated_pre_3= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =18)
        self.leaflet_anterior_rotated_intra_3 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =19)
        self.leaflet_posterior_parallel_pre_3= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =20)
        self.leaflet_posterior_parallel_intra_3 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =21)
        self.leaflet_posterior_rotated_pre_3= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =22)
        self.leaflet_posterior_rotated_intra_3 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =23)
        
        
        self.group_all_graphs()

        self.mpr_plot.addItem(self.leaflet_anterior_parallel_pre_1)
        self.mpr_plot.addItem(self.leaflet_anterior_parallel_intra_1)
        self.mpr_plot.addItem(self.leaflet_anterior_rotated_pre_1)
        self.mpr_plot.addItem(self.leaflet_anterior_rotated_intra_1)
        self.mpr_plot.addItem(self.leaflet_posterior_parallel_pre_1)
        self.mpr_plot.addItem(self.leaflet_posterior_parallel_intra_1)
        self.mpr_plot.addItem(self.leaflet_posterior_rotated_pre_1)
        self.mpr_plot.addItem(self.leaflet_posterior_rotated_intra_1)
        
        self.mpr_plot.addItem(self.leaflet_anterior_parallel_pre_2)
        self.mpr_plot.addItem(self.leaflet_anterior_parallel_intra_2)
        self.mpr_plot.addItem(self.leaflet_anterior_rotated_pre_2)
        self.mpr_plot.addItem(self.leaflet_anterior_rotated_intra_2)
        self.mpr_plot.addItem(self.leaflet_posterior_parallel_pre_2)
        self.mpr_plot.addItem(self.leaflet_posterior_parallel_intra_2)
        self.mpr_plot.addItem(self.leaflet_posterior_rotated_pre_2)
        self.mpr_plot.addItem(self.leaflet_posterior_rotated_intra_2)
        
        self.mpr_plot.addItem(self.leaflet_anterior_parallel_pre_3)
        self.mpr_plot.addItem(self.leaflet_anterior_parallel_intra_3)
        self.mpr_plot.addItem(self.leaflet_anterior_rotated_pre_3)
        self.mpr_plot.addItem(self.leaflet_anterior_rotated_intra_3)
        self.mpr_plot.addItem(self.leaflet_posterior_parallel_pre_3)
        self.mpr_plot.addItem(self.leaflet_posterior_parallel_intra_3)
        self.mpr_plot.addItem(self.leaflet_posterior_rotated_pre_3)
        self.mpr_plot.addItem(self.leaflet_posterior_rotated_intra_3)

        [graph.hide() for graph in self.all_graphs] 

        try:
            self.mpr_plot.sceneObj.sigMouseClicked.disconnect()
        except:
            pass
        self.mpr_plot.sceneObj.sigMouseClicked.connect(self.mouse_clicked)

        window_layout = QVBoxLayout()
        window_layout.addLayout(self.plot_layout)
        window_layout.addLayout(self.button_layout)

        self.setLayout(window_layout)

    def group_all_graphs(self):
        self.all_graphs = [self.leaflet_anterior_parallel_pre_1, self.leaflet_anterior_parallel_intra_1,
                           self.leaflet_anterior_rotated_pre_1, self.leaflet_anterior_rotated_intra_1,
                           self.leaflet_posterior_parallel_pre_1, self.leaflet_posterior_parallel_intra_1,
                           self.leaflet_posterior_rotated_pre_1, self.leaflet_posterior_rotated_intra_1,
                            
                           self.leaflet_anterior_parallel_pre_2, self.leaflet_anterior_parallel_intra_2,
                           self.leaflet_anterior_rotated_pre_2, self.leaflet_anterior_rotated_intra_2,
                           self.leaflet_posterior_parallel_pre_2, self.leaflet_posterior_parallel_intra_2,
                           self.leaflet_posterior_rotated_pre_2, self.leaflet_posterior_rotated_intra_2,
        
                           self.leaflet_anterior_parallel_pre_3, self.leaflet_anterior_parallel_intra_3,
                           self.leaflet_anterior_rotated_pre_3, self.leaflet_anterior_rotated_intra_3,
                           self.leaflet_posterior_parallel_pre_3, self.leaflet_posterior_parallel_intra_3,
                           self.leaflet_posterior_rotated_pre_3, self.leaflet_posterior_rotated_intra_3]

    def update_mpr(self):
        try:
            self.remove_annulus_intersections()
        except:
            pass
        image = self.current_mpr_array[self.current_mpr]
        self.mpr_image.setImage(image) 
        self.mpr_image.setLevels([0,255])
        self.add_annulus_intersections()
        self.show_and_hide_graphs()


    def mouse_clicked(self, ev):
        vb = self.mpr_plot.getPlotItem().getViewBox()
       
        scene_coords = ev.scenePos()
        self.mouse_pos_in_window= vb.mapSceneToView(scene_coords)

        new_point_x = self.mouse_pos_in_window.x()
        new_point_y = self.mouse_pos_in_window.y()

        self.pass_point_to_spline(new_point_x, new_point_y)

    def select_current_spline(self):
        if self.current_mpr == 0:
            if self.mode_anterior_posterior == 'ant':
                if self.mode_parallel ==True: 
                    if self.pre_mpr== True:
                        polygon_graph = self.leaflet_anterior_parallel_pre_1
                    if self.pre_mpr== False:
                        polygon_graph = self.leaflet_anterior_parallel_intra_1
                if self.mode_parallel ==False: 
                    if self.pre_mpr== True:
                        polygon_graph = self.leaflet_anterior_rotated_pre_1
                    if self.pre_mpr== False:
                        polygon_graph = self.leaflet_anterior_rotated_intra_1
            if self.mode_anterior_posterior == 'post':
                if self.mode_parallel ==True: 
                    if self.pre_mpr== True:
                        polygon_graph = self.leaflet_posterior_parallel_pre_1
                    if self.pre_mpr== False:
                        polygon_graph = self.leaflet_posterior_parallel_intra_1
                if self.mode_parallel ==False:
                    if self.pre_mpr== True:
                        polygon_graph = self.leaflet_posterior_rotated_pre_1
                    if self.pre_mpr== False:
                        polygon_graph = self.leaflet_posterior_rotated_intra_1
        if self.current_mpr == 1:
            if self.mode_anterior_posterior == 'ant':
                if self.mode_parallel ==True:
                    if self.pre_mpr== True: 
                        polygon_graph = self.leaflet_anterior_parallel_pre_2
                    if self.pre_mpr== False:
                        polygon_graph = self.leaflet_anterior_parallel_intra_2
                if self.mode_parallel ==False:
                    if self.pre_mpr== True:
                        polygon_graph = self.leaflet_anterior_rotated_pre_2
                    if self.pre_mpr== False:
                        polygon_graph = self.leaflet_anterior_rotated_intra_2
            if self.mode_anterior_posterior == 'post':
                if self.mode_parallel ==True: 
                    if self.pre_mpr== True:
                        polygon_graph = self.leaflet_posterior_parallel_pre_2
                    if self.pre_mpr== False:
                        polygon_graph = self.leaflet_posterior_parallel_intra_2
                if self.mode_parallel ==False:
                    if self.pre_mpr== True:
                        polygon_graph = self.leaflet_posterior_rotated_pre_2
                    if self.pre_mpr== False:
                        polygon_graph = self.leaflet_posterior_rotated_intra_2
        if self.current_mpr == 2:    
            if self.mode_anterior_posterior == 'ant':
                if self.mode_parallel ==True: 
                    if self.pre_mpr== True:
                        polygon_graph = self.leaflet_anterior_parallel_pre_3
                    if self.pre_mpr== False:
                        polygon_graph = self.leaflet_anterior_parallel_intra_3
                if self.mode_parallel ==False:
                    if self.pre_mpr== True:
                        polygon_graph = self.leaflet_anterior_rotated_pre_3
                    if self.pre_mpr== False:
                        polygon_graph = self.leaflet_anterior_rotated_intra_3
            if self.mode_anterior_posterior == 'post':
                if self.mode_parallel ==True: 
                    if self.pre_mpr== True:
                        polygon_graph = self.leaflet_posterior_parallel_pre_3
                    if self.pre_mpr== False:
                        polygon_graph = self.leaflet_posterior_parallel_intra_3
                if self.mode_parallel ==False:
                    if self.pre_mpr== True:
                        polygon_graph = self.leaflet_posterior_rotated_pre_3
                    if self.pre_mpr== False:
                        polygon_graph = self.leaflet_posterior_rotated_intra_3
        return polygon_graph
    
    def reset_current_spline(self):
        if self.current_mpr == 0:
            if self.mode_anterior_posterior == 'ant':
                if self.mode_parallel ==True: 
                    if self.pre_mpr== True:
                        self.leaflet_anterior_parallel_pre_1= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =0)
                    if self.pre_mpr== False:
                        self.leaflet_anterior_parallel_intra_1 = None
                        self.leaflet_anterior_parallel_intra_1 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =1)
                if self.mode_parallel ==False: 
                    if self.pre_mpr== True:
                        self.leaflet_anterior_rotated_pre_1= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =2)
                    if self.pre_mpr== False:
                        self.leaflet_anterior_rotated_intra_1 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =3)
            if self.mode_anterior_posterior == 'post':
                if self.mode_parallel ==True: 
                    if self.pre_mpr== True:
                        self.leaflet_posterior_parallel_pre_1= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =4)
                    if self.pre_mpr== False:
                        self.leaflet_posterior_parallel_intra_1 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =5)
                if self.mode_parallel ==False:
                    if self.pre_mpr== True:
                        self.leaflet_posterior_rotated_pre_1= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =6)
                    if self.pre_mpr== False:
                        self.leaflet_posterior_rotated_intra_1 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =7)
        if self.current_mpr == 1:
            if self.mode_anterior_posterior == 'ant':
                if self.mode_parallel ==True:
                    if self.pre_mpr== True: 
                        self.leaflet_anterior_parallel_pre_2= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =8)
                    if self.pre_mpr== False:
                        self.leaflet_anterior_parallel_intra_2 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =9)
                if self.mode_parallel ==False:
                    if self.pre_mpr== True:
                        self.leaflet_anterior_rotated_pre_2= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =10)
                    if self.pre_mpr== False:
                        self.leaflet_anterior_rotated_intra_2 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =11)
            if self.mode_anterior_posterior == 'post':
                if self.mode_parallel ==True: 
                    if self.pre_mpr== True:
                        self.leaflet_posterior_parallel_pre_2= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =12)
                    if self.pre_mpr== False:
                        self.leaflet_posterior_parallel_intra_2 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =13)
                if self.mode_parallel ==False:
                    if self.pre_mpr== True:
                        self.leaflet_posterior_rotated_pre_2= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =14)
                    if self.pre_mpr== False:
                        self.leaflet_posterior_rotated_intra_2 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =15)
        if self.current_mpr == 2:    
            if self.mode_anterior_posterior == 'ant':
                if self.mode_parallel ==True: 
                    if self.pre_mpr== True:
                        self.leaflet_anterior_parallel_pre_3= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =16)
                    if self.pre_mpr== False:
                        self.leaflet_anterior_parallel_intra_3 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =17)
                if self.mode_parallel ==False:
                    if self.pre_mpr== True:
                        self.leaflet_anterior_rotated_pre_3= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =18)
                    if self.pre_mpr== False:
                        self.leaflet_anterior_rotated_intra_3 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_1, brush=pg.mkBrush("r"), id="pink", tab = self, graph_number =19)
            if self.mode_anterior_posterior == 'post':
                if self.mode_parallel ==True: 
                    if self.pre_mpr== True:
                        self.leaflet_posterior_parallel_pre_3= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =20)
                    if self.pre_mpr== False:
                        self.leaflet_posterior_parallel_intra_3 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =21)
                if self.mode_parallel ==False:
                    if self.pre_mpr== True:
                        self.leaflet_posterior_rotated_pre_3= Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =22)
                    if self.pre_mpr== False:
                        self.leaflet_posterior_rotated_intra_3 = Leaflet_Graph(self.mpr_plot, self.leaflet_pen_2, brush=pg.mkBrush("g"), id="pink", tab = self, graph_number =23)
      

    def pass_point_to_spline(self, new_point_x, new_point_y):
        polygon_graph = self.select_current_spline()
        polygon_array = polygon_graph.get_waypoints()
        self.mouse_pos_in_image = [int(new_point_x), int(new_point_y)]
        
        polygon_array[polygon_graph.nr_selected_points_for_polygon, :] = self.mouse_pos_in_image
        polygon_graph.nr_selected_points_for_polygon = polygon_graph.nr_selected_points_for_polygon+1

        self.animation.create_spline(polygon_array, polygon_graph)
        
        polygon_graph.waypoint_number = polygon_graph.waypoint_number +1

        return
    
    def show_and_hide_graphs(self):
        [graph.hide() for graph in self.all_graphs] 
        if self.current_mpr== 0:
            if self.mode_parallel == True:
                if self.pre_mpr == True:
                    self.leaflet_anterior_parallel_pre_1.show()
                    self.leaflet_posterior_parallel_pre_1.show()
                if self.pre_mpr == False:
                    self.leaflet_anterior_parallel_intra_1.show()
                    self.leaflet_posterior_parallel_intra_1.show()

            if self.mode_parallel == False:
                if self.pre_mpr == True:
                    self.leaflet_anterior_rotated_pre_1.show()
                    self.leaflet_posterior_rotated_pre_1.show()
                if self.pre_mpr == False:
                    self.leaflet_anterior_rotated_intra_1.show()
                    self.leaflet_posterior_rotated_intra_1.show()
        
        if self.current_mpr== 1:
            if self.mode_parallel == True:
                if self.pre_mpr == True:
                    self.leaflet_anterior_parallel_pre_2.show()
                    self.leaflet_posterior_parallel_pre_2.show()
                if self.pre_mpr == False:
                    self.leaflet_anterior_parallel_intra_2.show()
                    self.leaflet_posterior_parallel_intra_2.show()

            if self.mode_parallel == False:
                if self.pre_mpr == True:
                    self.leaflet_anterior_rotated_pre_2.show()
                    self.leaflet_posterior_rotated_pre_2.show()
                if self.pre_mpr == False:
                    self.leaflet_anterior_rotated_intra_2.show()
                    self.leaflet_posterior_rotated_intra_2.show()
        
        if self.current_mpr== 2:
            if self.mode_parallel == True:
                if self.pre_mpr == True:
                    self.leaflet_anterior_parallel_pre_3.show()
                    self.leaflet_posterior_parallel_pre_3.show()
                if self.pre_mpr == False:
                    self.leaflet_anterior_parallel_intra_3.show()
                    self.leaflet_posterior_parallel_intra_3.show()

            if self.mode_parallel == False:
                if self.pre_mpr == True:
                    self.leaflet_anterior_rotated_pre_3.show()
                    self.leaflet_posterior_rotated_pre_3.show()
                if self.pre_mpr == False:
                    self.leaflet_anterior_rotated_intra_3.show()
                    self.leaflet_posterior_rotated_intra_3.show()
        
    def delete_selected_spline_button_was_pressed(self):
        polygon_graph = self.select_current_spline()
        self.mpr_plot.removeItem(polygon_graph)
        self.reset_current_spline()
        polygon_graph = self.select_current_spline()
        self.mpr_plot.addItem(polygon_graph)
        self.group_all_graphs()

    def anterior_radio_button_was_pressed(self):
        self.mode_anterior_posterior = 'ant'

    def posterior_radio_button_was_pressed(self):
        self.mode_anterior_posterior = 'post'

    def next_button_was_pressed(self):
        self.current_mpr = self.current_mpr +1
        self.back_button.setEnabled(True)
        if self.current_mpr == 2:
            self.next_button.setEnabled(False)
        self.update_mpr()

    def back_button_was_pressed(self):
        self.current_mpr = self.current_mpr -1
        self.next_button.setEnabled(True)
        if self.current_mpr == 0:
            self.back_button.setEnabled(False)
        self.update_mpr()

    def add_annulus_intersections(self):
        if self.pre_mpr == True:
            if self.mode_parallel == True:
                if self.current_mpr == 0:
                    self.intersection = self.mpr_slicer.transform_world_to_mpr_spline(self.parallel_annulus_0_pre, self.transformation_array_parallel_pre[0][0], self.transformation_array_parallel_pre[0][1])
                if self.current_mpr == 1:
                    self.intersection = self.mpr_slicer.transform_world_to_mpr_spline(self.parallel_annulus_1_pre, self.transformation_array_parallel_pre[1][0], self.transformation_array_parallel_pre[1][1])
                if self.current_mpr == 2:
                    self.intersection = self.mpr_slicer.transform_world_to_mpr_spline(self.parallel_annulus_2_pre, self.transformation_array_parallel_pre[2][0], self.transformation_array_parallel_pre[2][1])
            if self.mode_parallel == False:
                if self.current_mpr == 0:
                    self.intersection = self.mpr_slicer.transform_world_to_mpr_spline(self.rotated_annulus_0_pre, self.transformation_array_rotated_pre[0][0], self.transformation_array_rotated_pre[0][1])
                if self.current_mpr == 1:
                    self.intersection = self.mpr_slicer.transform_world_to_mpr_spline(self.rotated_annulus_1_pre, self.transformation_array_rotated_pre[1][0], self.transformation_array_rotated_pre[1][1])
                if self.current_mpr == 2:
                    self.intersection = self.mpr_slicer.transform_world_to_mpr_spline(self.rotated_annulus_2_pre, self.transformation_array_rotated_pre[2][0], self.transformation_array_rotated_pre[2][1])
        if self.pre_mpr == False:
            if self.mode_parallel == True:
                if self.current_mpr == 0:
                    self.intersection = self.mpr_slicer.transform_world_to_mpr_spline(self.parallel_annulus_0_intra, self.transformation_array_parallel_intra[0][0], self.transformation_array_parallel_intra[0][1])
                if self.current_mpr == 1:
                    self.intersection = self.mpr_slicer.transform_world_to_mpr_spline(self.parallel_annulus_1_intra, self.transformation_array_parallel_intra[1][0], self.transformation_array_parallel_intra[1][1])
                if self.current_mpr == 2:
                    self.intersection = self.mpr_slicer.transform_world_to_mpr_spline(self.parallel_annulus_2_intra, self.transformation_array_parallel_intra[2][0], self.transformation_array_parallel_intra[2][1])
            if self.mode_parallel == False:
                if self.current_mpr == 0:
                    self.intersection = self.mpr_slicer.transform_world_to_mpr_spline(self.rotated_annulus_0_intra, self.transformation_array_rotated_intra[0][0], self.transformation_array_rotated_intra[0][1])
                if self.current_mpr == 1:
                    self.intersection = self.mpr_slicer.transform_world_to_mpr_spline(self.rotated_annulus_1_intra, self.transformation_array_rotated_intra[1][0], self.transformation_array_rotated_intra[1][1])
                if self.current_mpr == 2:
                    self.intersection = self.mpr_slicer.transform_world_to_mpr_spline(self.rotated_annulus_2_intra, self.transformation_array_rotated_intra[2][0], self.transformation_array_rotated_intra[2][1])
        x_1 = self.intersection[0,0]
        y_1 = self.intersection[0,1]
        x_2 = self.intersection[1,0]
        y_2 = self.intersection[1,1]
        print('self.intersection: ', self.intersection)
        
        self.intersection_1 = self.mpr_plot.plot([x_1], [y_1], symbol='x', symbolPen = self.orange_pen, symbolBrush=(255, 165, 0), symbolSize=12)
        self.intersection_2 = self.mpr_plot.plot([x_2], [y_2], symbol='x', symbolPen = self.orange_pen, symbolBrush=(255, 165, 0), symbolSize=12)

    def remove_annulus_intersections(self):
        self.mpr_plot.removeItem(self.intersection_1)
        self.mpr_plot.removeItem(self.intersection_2)

    def get_mprs_parallel_button_was_pressed(self):
        #mpr in normal direction = 0
        #mpr through clip = 1
        #mpr against normal direction = 2
        saved_state = self.mpr_slicer.pre_or_intra
        if self.mpr_slicer.pre_or_intra == 'pre':
            self.mpr_slicer.change_volume_mode()
            self.mpr_slicer.set_volume()
            self.mpr_slicer.set_mv_model()
            self.mpr_slicer.define_anatomic_coordi(self.mpr_slicer.annulus_points)
            self.mpr_slicer.prepare_visualization(False) 

        mpr0_intra, mpr1_intra, mpr2_intra, t_point_0_intra, t_point_1_intra, t_point_2_intra, coordi_0_intra, coordi_1_intra, coordi_2_intra, saved_points_0_intra, saved_points_1_intra, saved_points_2_intra, self.parallel_annulus_0_intra, self.parallel_annulus_1_intra, self.parallel_annulus_2_intra = self.mpr_slicer.calculate_parallel_intra_model_mprs()
        self.mpr_array_parallel_intra = [mpr0_intra, mpr1_intra, mpr2_intra]
        self.transformation_array_parallel_intra = [[t_point_0_intra, coordi_0_intra],[t_point_1_intra, coordi_1_intra],[t_point_2_intra, coordi_2_intra]]
        self.update_current_mpr_array()
        self.update_mpr()

        #Get the model for pre data
        self.mpr_slicer.change_volume_mode()
        self.mpr_slicer.set_volume()
        self.mpr_slicer.set_mv_model()
        self.mpr_slicer.define_anatomic_coordi(self.mpr_slicer.annulus_points)
        self.mpr_slicer.prepare_visualization(False) 
        
        mpr0_pre, mpr1_pre, mpr2_pre, t_point_0_pre, t_point_1_pre, t_point_2_pre, coordi_0_pre, coordi_1_pre, coordi_2_pre, self.parallel_annulus_0_pre, self.parallel_annulus_1_pre, self.parallel_annulus_2_pre = self.mpr_slicer.calculate_pre_model_mprs(saved_points_0_intra, saved_points_1_intra, saved_points_2_intra)
        self.mpr_array_parallel_pre = [mpr0_pre, mpr1_pre, mpr2_pre]
        self.transformation_array_parallel_pre = [[t_point_0_pre, coordi_0_pre],[t_point_1_pre, coordi_1_pre],[t_point_2_pre, coordi_2_pre]]

        if saved_state != self.mpr_slicer.pre_or_intra:
            self.mpr_slicer.change_volume_mode()
            self.mpr_slicer.set_volume()
            self.mpr_slicer.set_mv_model()
            self.mpr_slicer.define_anatomic_coordi(self.mpr_slicer.annulus_points)
            self.mpr_slicer.prepare_visualization(False) 

        

    def get_mprs_rotated_button_was_pressed(self):
        #mpr in normal direction = 0
        #mpr through clip = 1
        #mpr against normal direction = 2
        saved_state = self.mpr_slicer.pre_or_intra
        if self.mpr_slicer.pre_or_intra == 'pre':
            self.mpr_slicer.change_volume_mode()
            self.mpr_slicer.set_volume()
            self.mpr_slicer.set_mv_model()
            self.mpr_slicer.define_anatomic_coordi(self.mpr_slicer.annulus_points)
            self.mpr_slicer.prepare_visualization(False) 

        mpr0_intra, mpr1_intra, mpr2_intra, t_point_0_intra, t_point_1_intra, t_point_2_intra, coordi_0_intra, coordi_1_intra, coordi_2_intra, saved_points_0_intra, saved_points_1_intra, saved_points_2_intra, self.rotated_annulus_0_intra, self.rotated_annulus_1_intra, self.rotated_annulus_2_intra = self.mpr_slicer.calculate_rotated_intra_model_mprs()
        self.mpr_array_rotated_intra = [mpr0_intra, mpr1_intra, mpr2_intra]
        self.transformation_array_rotated_intra = [[t_point_0_intra, coordi_0_intra],[t_point_1_intra, coordi_1_intra],[t_point_2_intra, coordi_2_intra]]
        self.update_current_mpr_array()
        self.update_mpr()

        #Get the model for pre data
        self.mpr_slicer.change_volume_mode()
        self.mpr_slicer.set_volume()
        self.mpr_slicer.set_mv_model()
        self.mpr_slicer.define_anatomic_coordi(self.mpr_slicer.annulus_points)
        self.mpr_slicer.prepare_visualization(False) 
        
        mpr0_pre, mpr1_pre, mpr2_pre, t_point_0_pre, t_point_1_pre, t_point_2_pre, coordi_0_pre, coordi_1_pre, coordi_2_pre, self.rotated_annulus_0_pre, self.rotated_annulus_1_pre, self.rotated_annulus_2_pre = self.mpr_slicer.calculate_pre_model_mprs(saved_points_0_intra, saved_points_1_intra, saved_points_2_intra)
        self.mpr_array_rotated_pre = [mpr0_pre, mpr1_pre, mpr2_pre]
        self.transformation_array_rotated_pre = [[t_point_0_pre, coordi_0_pre],[t_point_1_pre, coordi_1_pre],[t_point_2_pre, coordi_2_pre]]

        if saved_state != self.mpr_slicer.pre_or_intra:
            self.mpr_slicer.change_volume_mode()
            self.mpr_slicer.set_volume()
            self.mpr_slicer.set_mv_model()
            self.mpr_slicer.define_anatomic_coordi(self.mpr_slicer.annulus_points)
            self.mpr_slicer.prepare_visualization(False) 
        

    def change_pre_intra_button_was_pressed(self):
        self.pre_mpr = not self.pre_mpr
        
        if self.pre_mpr == True:
            self.change_pre_intra_button.setText('Change to intra')
            if self.mode_parallel == True:
                self.current_mpr_array = self.mpr_array_parallel_pre
            if self.mode_parallel == False:
                self.current_mpr_array = self.mpr_array_rotated_pre
        if self.pre_mpr == False:
            self.change_pre_intra_button.setText('Change to pre')
            if self.mode_parallel == True:
                self.current_mpr_array = self.mpr_array_parallel_intra
            if self.mode_parallel == False:
                self.current_mpr_array = self.mpr_array_rotated_intra
        self.update_mpr()

    def update_current_mpr_array(self):
        if self.pre_mpr == True:
            if self.mode_parallel == True:
                self.current_mpr_array = self.mpr_array_parallel_pre
            if self.mode_parallel == False:
                self.current_mpr_array = self.mpr_array_rotated_pre
        if self.pre_mpr == False:
            if self.mode_parallel == True:
                self.current_mpr_array = self.mpr_array_parallel_intra
            if self.mode_parallel == False:
                self.current_mpr_array = self.mpr_array_rotated_intra


    def change_mode_button_was_pressed(self):
        self.mode_parallel = not self.mode_parallel
        if self.mode_parallel == True:
            self.change_mode_button.setText('Change to rotated')
            if self.pre_mpr == True:
                self.current_mpr_array = self.mpr_array_parallel_pre
            if self.pre_mpr == False:
                self.current_mpr_array = self.mpr_array_parallel_intra
            
        if self.mode_parallel == False:
            self.change_mode_button.setText('Change to parallel')
            if self.pre_mpr == True:
                self.current_mpr_array = self.mpr_array_rotated_pre
            if self.pre_mpr == False:
                self.current_mpr_array = self.mpr_array_rotated_intra
            
        self.update_mpr()
        
        

    def create_model_parallel_pre_button_was_pressed(self):
        pts_ant_p_p_1 = self.leaflet_anterior_parallel_pre_1.get_spline_points()
        pts_ant_p_p_2 = self.leaflet_anterior_parallel_pre_2.get_spline_points()
        pts_ant_p_p_3 = self.leaflet_anterior_parallel_pre_3.get_spline_points()

        x_ant_p_p_1, y_ant_p_p_1, z_ant_p_p_1 = self.calculate_u_v_spline(pts_ant_p_p_1, self.transformation_array_parallel_pre[0])
        x_ant_p_p_2, y_ant_p_p_2, z_ant_p_p_2 = self.calculate_u_v_spline(pts_ant_p_p_2, self.transformation_array_parallel_pre[1])
        x_ant_p_p_3, y_ant_p_p_3, z_ant_p_p_3 = self.calculate_u_v_spline(pts_ant_p_p_3, self.transformation_array_parallel_pre[2])

        ant_p_p_1 = np.column_stack((x_ant_p_p_1, y_ant_p_p_1, z_ant_p_p_1))
        ant_p_p_2 = np.column_stack((x_ant_p_p_2, y_ant_p_p_2, z_ant_p_p_2))
        ant_p_p_3 = np.column_stack((x_ant_p_p_3, y_ant_p_p_3, z_ant_p_p_3))
        self.ant_p_p_vertices, self.ant_p_p_faces = self.create_thin_plate_spline(ant_p_p_1, ant_p_p_2, ant_p_p_3)


        pts_post_p_p_1 = self.leaflet_posterior_parallel_pre_1.get_spline_points()
        pts_post_p_p_2 = self.leaflet_posterior_parallel_pre_2.get_spline_points()
        pts_post_p_p_3 = self.leaflet_posterior_parallel_pre_3.get_spline_points()

        x_post_p_p_1, y_post_p_p_1, z_post_p_p_1 = self.calculate_u_v_spline(pts_post_p_p_1, self.transformation_array_parallel_pre[0])
        x_post_p_p_2, y_post_p_p_2, z_post_p_p_2 = self.calculate_u_v_spline(pts_post_p_p_2, self.transformation_array_parallel_pre[1])
        x_post_p_p_3, y_post_p_p_3, z_post_p_p_3 = self.calculate_u_v_spline(pts_post_p_p_3, self.transformation_array_parallel_pre[2])

        post_p_p_1 = np.column_stack((x_post_p_p_1, y_post_p_p_1, z_post_p_p_1))
        post_p_p_2 = np.column_stack((x_post_p_p_2, y_post_p_p_2, z_post_p_p_2))
        post_p_p_3 = np.column_stack((x_post_p_p_3, y_post_p_p_3, z_post_p_p_3))
        self.post_p_p_vertices, self.post_p_p_faces = self.create_thin_plate_spline(post_p_p_1, post_p_p_2, post_p_p_3)

    def create_model_rotated_pre_button_was_pressed(self):
        pts_ant_r_p_1 = self.leaflet_anterior_rotated_pre_1.get_spline_points()
        pts_ant_r_p_2 = self.leaflet_anterior_rotated_pre_2.get_spline_points()
        pts_ant_r_p_3 = self.leaflet_anterior_rotated_pre_3.get_spline_points()

        x_ant_r_p_1, y_ant_r_p_1, z_ant_r_p_1 = self.calculate_u_v_spline(pts_ant_r_p_1, self.transformation_array_rotated_pre[0])
        x_ant_r_p_2, y_ant_r_p_2, z_ant_r_p_2 = self.calculate_u_v_spline(pts_ant_r_p_2, self.transformation_array_rotated_pre[1])
        x_ant_r_p_3, y_ant_r_p_3, z_ant_r_p_3 = self.calculate_u_v_spline(pts_ant_r_p_3, self.transformation_array_rotated_pre[2])

        ant_r_p_1 = np.column_stack((x_ant_r_p_1, y_ant_r_p_1, z_ant_r_p_1))
        ant_r_p_2 = np.column_stack((x_ant_r_p_2, y_ant_r_p_2, z_ant_r_p_2))
        ant_r_p_3 = np.column_stack((x_ant_r_p_3, y_ant_r_p_3, z_ant_r_p_3))
        self.ant_r_p_vertices, self.ant_r_p_faces = self.create_thin_plate_spline(ant_r_p_1, ant_r_p_2, ant_r_p_3)


        pts_post_r_p_1 = self.leaflet_posterior_rotated_pre_1.get_spline_points()
        pts_post_r_p_2 = self.leaflet_posterior_rotated_pre_2.get_spline_points()
        pts_post_r_p_3 = self.leaflet_posterior_rotated_pre_3.get_spline_points()

        x_post_r_p_1, y_post_r_p_1, z_post_r_p_1 = self.calculate_u_v_spline(pts_post_r_p_1, self.transformation_array_rotated_pre[0])
        x_post_r_p_2, y_post_r_p_2, z_post_r_p_2 = self.calculate_u_v_spline(pts_post_r_p_2, self.transformation_array_rotated_pre[1])
        x_post_r_p_3, y_post_r_p_3, z_post_r_p_3 = self.calculate_u_v_spline(pts_post_r_p_3, self.transformation_array_rotated_pre[2])

        post_r_p_1 = np.column_stack((x_post_r_p_1, y_post_r_p_1, z_post_r_p_1))
        post_r_p_2 = np.column_stack((x_post_r_p_2, y_post_r_p_2, z_post_r_p_2))
        post_r_p_3 = np.column_stack((x_post_r_p_3, y_post_r_p_3, z_post_r_p_3))
        self.post_r_p_vertices, self.post_r_p_faces = self.create_thin_plate_spline(post_r_p_1, post_r_p_2, post_r_p_3)

    def create_model_parallel_intra_button_was_pressed(self):
        pts_ant_p_i_1 = self.leaflet_anterior_parallel_intra_1.get_spline_points()
        pts_ant_p_i_2 = self.leaflet_anterior_parallel_intra_2.get_spline_points()
        pts_ant_p_i_3 = self.leaflet_anterior_parallel_intra_3.get_spline_points()

        x_ant_p_i_1, y_ant_p_i_1, z_ant_p_i_1 = self.calculate_u_v_spline(pts_ant_p_i_1, self.transformation_array_parallel_intra[0])
        x_ant_p_i_2, y_ant_p_i_2, z_ant_p_i_2 = self.calculate_u_v_spline(pts_ant_p_i_2, self.transformation_array_parallel_intra[1])
        x_ant_p_i_3, y_ant_p_i_3, z_ant_p_i_3 = self.calculate_u_v_spline(pts_ant_p_i_3, self.transformation_array_parallel_intra[2])

        ant_p_i_1 = np.column_stack((x_ant_p_i_1, y_ant_p_i_1, z_ant_p_i_1))
        ant_p_i_2 = np.column_stack((x_ant_p_i_2, y_ant_p_i_2, z_ant_p_i_2))
        ant_p_i_3 = np.column_stack((x_ant_p_i_3, y_ant_p_i_3, z_ant_p_i_3))
        self.ant_p_i_vertices, self.ant_p_i_faces = self.create_thin_plate_spline(ant_p_i_1, ant_p_i_2, ant_p_i_3)


        pts_post_p_i_1 = self.leaflet_posterior_parallel_intra_1.get_spline_points()
        pts_post_p_i_2 = self.leaflet_posterior_parallel_intra_2.get_spline_points()
        pts_post_p_i_3 = self.leaflet_posterior_parallel_intra_3.get_spline_points()

        x_post_p_i_1, y_post_p_i_1, z_post_p_i_1 = self.calculate_u_v_spline(pts_post_p_i_1, self.transformation_array_parallel_intra[0])
        x_post_p_i_2, y_post_p_i_2, z_post_p_i_2 = self.calculate_u_v_spline(pts_post_p_i_2, self.transformation_array_parallel_intra[1])
        x_post_p_i_3, y_post_p_i_3, z_post_p_i_3 = self.calculate_u_v_spline(pts_post_p_i_3, self.transformation_array_parallel_intra[2])

        post_p_i_1 = np.column_stack((x_post_p_i_1, y_post_p_i_1, z_post_p_i_1))
        post_p_i_2 = np.column_stack((x_post_p_i_2, y_post_p_i_2, z_post_p_i_2))
        post_p_i_3 = np.column_stack((x_post_p_i_3, y_post_p_i_3, z_post_p_i_3))
        self.post_p_i_vertices, self.post_p_i_faces = self.create_thin_plate_spline(post_p_i_1, post_p_i_2, post_p_i_3)

    def create_model_rotated_intra_button_was_pressed(self):
        pts_ant_r_i_1 = self.leaflet_anterior_rotated_intra_1.get_spline_points()
        pts_ant_r_i_2 = self.leaflet_anterior_rotated_intra_2.get_spline_points()
        pts_ant_r_i_3 = self.leaflet_anterior_rotated_intra_3.get_spline_points()

        x_ant_r_i_1, y_ant_r_i_1, z_ant_r_i_1 = self.calculate_u_v_spline(pts_ant_r_i_1, self.transformation_array_rotated_intra[0])
        x_ant_r_i_2, y_ant_r_i_2, z_ant_r_i_2 = self.calculate_u_v_spline(pts_ant_r_i_2, self.transformation_array_rotated_intra[1])
        x_ant_r_i_3, y_ant_r_i_3, z_ant_r_i_3 = self.calculate_u_v_spline(pts_ant_r_i_3, self.transformation_array_rotated_intra[2])

        ant_r_i_1 = np.column_stack((x_ant_r_i_1, y_ant_r_i_1, z_ant_r_i_1))
        ant_r_i_2 = np.column_stack((x_ant_r_i_2, y_ant_r_i_2, z_ant_r_i_2))
        ant_r_i_3 = np.column_stack((x_ant_r_i_3, y_ant_r_i_3, z_ant_r_i_3))
        self.ant_r_i_vertices, self.ant_r_i_faces = self.create_thin_plate_spline(ant_r_i_1, ant_r_i_2, ant_r_i_3)


        pts_post_r_i_1 = self.leaflet_posterior_rotated_intra_1.get_spline_points()
        pts_post_r_i_2 = self.leaflet_posterior_rotated_intra_2.get_spline_points()
        pts_post_r_i_3 = self.leaflet_posterior_rotated_intra_3.get_spline_points()

        x_post_r_i_1, y_post_r_i_1, z_post_r_i_1 = self.calculate_u_v_spline(pts_post_r_i_1, self.transformation_array_rotated_intra[0])
        x_post_r_i_2, y_post_r_i_2, z_post_r_i_2 = self.calculate_u_v_spline(pts_post_r_i_2, self.transformation_array_rotated_intra[1])
        x_post_r_i_3, y_post_r_i_3, z_post_r_i_3 = self.calculate_u_v_spline(pts_post_r_i_3, self.transformation_array_rotated_intra[2])

        post_r_i_1 = np.column_stack((x_post_r_i_1, y_post_r_i_1, z_post_r_i_1))
        post_r_i_2 = np.column_stack((x_post_r_i_2, y_post_r_i_2, z_post_r_i_2))
        post_r_i_3 = np.column_stack((x_post_r_i_3, y_post_r_i_3, z_post_r_i_3))
        self.post_r_i_vertices, self.post_r_i_faces = self.create_thin_plate_spline(post_r_i_1, post_r_i_2, post_r_i_3)


    def calculate_spline_param(self, way_points):
        single_curve_lengths = np.zeros(way_points.shape[0])
        for i in range(way_points.shape[0]-1):
            single_curve_lengths[i+1] = np.linalg.norm(way_points[i,:]-way_points[i+1,:])
        total_curve_length = np.cumsum(single_curve_lengths)
        t = total_curve_length/(total_curve_length[-1])     

        return t
    
    def calculate_u_v_spline(self, way_points_2d, transformation_set):
        #in 3D
        way_points_2d = np.c_[way_points_2d, np.zeros((way_points_2d.shape[0], 1))]
        way_points = np.zeros_like(way_points_2d)
        for i in range(way_points_2d.shape[0]):
            way_points[i,:]= self.mpr_slicer.transform_mpr_spline_to_world(way_points_2d[i,:], transformation_set[0], transformation_set[1])
        
        self.spline_3D= Tube(way_points, r=1.0, cap=True, res=12, c='blue', alpha=1.0)
        self.spline_3D.scale(s = self.mpr_slicer.slice_thickness_intra)
        self.mpr_slicer.plot.add(self.spline_3D)
        
        u = self.calculate_spline_param(way_points)
        
        fx_cs = CubicSpline(u, way_points[:,0])
        fy_cs = CubicSpline(u, way_points[:,1])
        fz_cs = CubicSpline(u, way_points[:,2])
        
        var = np.linspace(0, 1, num=100)
        x_cs = fx_cs(var)
        y_cs = fy_cs(var)
        z_cs = fz_cs(var)

        tube_points = np.column_stack((x_cs, y_cs, z_cs))
        self.spline_3D_22= Tube(tube_points, r=1.0, cap=True, res=12, c='pink', alpha=1.0)
        self.spline_3D_22.scale(s = self.mpr_slicer.slice_thickness_intra)
        self.mpr_slicer.plot.add(self.spline_3D_22)
        return x_cs, y_cs, z_cs
    
    def create_thin_plate_spline(self, spline_1, spline_2, spline_3):

        all_triangles = np.zeros(((spline_1.shape[0]-1)*4,3))

        len_spline = spline_1.shape[0]
        j = 0
        for i in range(len_spline-1):
            all_triangles[j,:] = [i, i+1, len_spline +i]
            all_triangles[j+1,:] = [i+1, len_spline+i+1, len_spline+i]
            j = j+2
        for i in range(len_spline-1):
            all_triangles[j,:] = [len_spline+i, len_spline+i+1, 2*len_spline+i]
            all_triangles[j+1,:] = [len_spline+i+1, 2*len_spline+i+1, 2*len_spline+i]
            j = j+2
        

        vertices = np.concatenate((spline_1, spline_2, spline_3), axis= 0)
        faces = all_triangles.astype(np.int16)
        return vertices, faces