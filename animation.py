import sys

from pydicom import dcmread
from PIL import ImageTk, Image, ImageQt

import numpy as np

import matplotlib.animation as plt_animation
import matplotlib as plt
import matplotlib.pyplot as plt_py
from pydicom.fileset import FileSet
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer,QDateTime, Qt, Slot,QPointF, Signal
from PySide6.QtGui import QAction, QPainter,QImage, QPixmap, QPen, QIntValidator, QColor, QKeySequence, QPolygon, QPolygonF, QBrush, QMouseEvent, QFont, QCursor
from PySide6.QtWidgets import (QApplication, QLabel,QMainWindow, QPushButton, QWidget, QListWidget,
                               QLineEdit,QFileDialog,QVBoxLayout, QHBoxLayout,QDialogButtonBox,
                               QGridLayout, QCheckBox, QMessageBox, QSizePolicy, QSlider, QGraphicsView, 
                               QGraphicsScene, QStyle,QSpacerItem, QRadioButton)
from random import randint, choice
import matplotlib.patches as mpatches
import pyqtgraph as pg
from scipy.interpolate import CubicSpline, UnivariateSpline, LSQUnivariateSpline



class Animation():
    def __init__(self, data, dicom_information):
        self.data = data
        self.dicom_information = dicom_information
        self.number_of_frames= self.data.shape[0] #number of frames that are available
        self.frame = 0 #number of frame that is currently shown in the app
        self.pix = None

        self.mpr_image = None


        #timer for the image sequence 
        self.time = 0
        self.Timer_on =False
        self.timer=QTimer()

        
        self.leaflet_nr = 0


 

    def update_figure(self):
        #displayes the current frame in the gui and adds a scale
        #get the image
        im = self.get_frame()
        im = np.flipud(im)

        

    def calculate_scale(self):
        #determine how long (=how many pixel) the scale in the gui should be
        physical_delta_x = self.dicom_information.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
        # number of pixel * physical delta = distance in cm
        scale_in_pixel = 1 / physical_delta_x 
        #start point of the scale
        x1 = 600 
        #end point of the scale
        x2 = x1 + scale_in_pixel
        return x1, x2


    def frame_animation_timer(self):
        if self.Timer_on:
            self.timer.stop()
            self.Timer_on = False
        else:
            self.timer.start(1000)
            self.Timer_on = True

    def start_frame_sequence(self):
        self.frame = self.frame +1
        if self.frame == 233:
            self.frame =0
        self.update_figure()

        
    def get_frame(self):
        #gets current frame (not necessarily the on that is displayed in the gui)
        return self.data[self.frame]

    def go_to_last_frame(self):
        self.timer.stop()
        self.Timer_on = False
        self.frame = self.frame -1
        if self.frame == -1:
            self.frame = self.number_of_frames-1
        self.update_figure()
        
    def go_to_next_frame(self):
        self.timer.stop()
        self.Timer_on = False
        self.frame = self.frame +1
        if self.frame == self.number_of_frames:
            self.frame = 0
        self.update_figure()


    def create_spline(self, waypoint_array, graph):
        
        #delete zero rows and sort the array by x value
        #point_array= point_array[point_array[:, 0].argsort()], das führt dazu dass man zB keine C Formen machen kann
        waypoint_array = waypoint_array[~np.all(waypoint_array == 0, axis=1)]

        if waypoint_array.shape[0] > 1:
            waypoint_array = np.reshape(waypoint_array, (-1, 2))

            
            x_spline, y_spline = graph.calculate_cubic_spline(np.transpose(waypoint_array))
            p = np.column_stack((x_spline, y_spline))
            graph.setData(pos = p)

        #übergibt nur die Punkte an den Graphen, ohne den Spline    
        else: 
            graph.setData(pos = waypoint_array)

 


