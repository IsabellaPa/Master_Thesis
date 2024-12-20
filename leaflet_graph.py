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


class Leaflet_Graph(pg.GraphItem):
    """Graph item for draggable points."""
    PointChanged = Signal(int)
    def __init__(self, plot, pen, brush, id, tab, graph_number): 
        self.drag_point = None
        self.drag_offset = None
        self.id = id
        self.Test=False
        self.waypoints = np.zeros((100,2))
        self.new_waypoint_array = None
        self.nr_selected_points_for_polygon = 0
        self.step = 10
        self.length_of_leaflet = 0
        self.data_points = None
        self.plot = plot
        self.graph_number = graph_number
        self.tab = tab
        self.waypoint_number = 0

        self.leaflet_scaling = [1, 1]

        pg.GraphItem.__init__(self, symbolBrush=brush)

    def setData(self, **kwds):
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['adj'] = np.column_stack((np.arange(0, npts-1), np.arange(1, npts)))
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
            self.data_points = self.data['pos']
        self.updateGraph()

    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)



    def mouseDragEvent(self, ev):
        
        if self.Test==True or self.Test==False: 
            if ev.button() != Qt.LeftButton:
                ev.ignore()
                return

            if ev.isStart():
                pos = ev.buttonDownPos()
                pts = self.scatter.pointsAt(pos)
                if len(pts) == 0:
                    ev.ignore()
                    return
                self.dragPoint = pts[0]
                ind = pts[0].data()[0]
                
                self.dragOffset_x = self.data['pos'][ind][0] - pos[0]
                self.dragOffset_y = self.data['pos'][ind][1] - pos[1]

            elif ev.isFinish():
                self.set_waypoints(np.transpose(self.new_waypoint_array))
                self.draw_new_waypoint(self.new_waypoint)
                self.waypoint_number = self.waypoint_number + 1
                try:
                    self.update_length_label()
                except:
                    pass
                self.dragPoint = None
                return
            else:
                if self.dragPoint is None:
                    ev.ignore()
                    return
            
            ind = self.dragPoint.data()[0]
           
            self.new_waypoint = [ev.pos()[0] + self.dragOffset_x, ev.pos()[1] + self.dragOffset_y]
            self.new_waypoint_array = self.add_new_spline_waypoint(np.round(self.new_waypoint), ind)
            x_spline, y_spline = self.calculate_cubic_spline(self.new_waypoint_array)
            p = np.column_stack((x_spline, y_spline))
            self.setData(pos=p)
            

            


            self.updateGraph()
            ev.accept()

            
            return
        else:
            return
                
    def calculate_length_in_mm(self, points_unscaled):
        unscaled_x = points_unscaled[0,:]
        unscaled_y = points_unscaled[1,:]

        scaled_x = unscaled_x * self.leaflet_scaling[0]
        scaled_y = unscaled_y * self.leaflet_scaling[1]

        scaled_points= np.array([scaled_x, scaled_y])

        return scaled_points
    
    def update_length_label(self):
        self.tab.update_leaflet_length_label(self.graph_number)
        
        
    def calculate_spline_param(self, way_points):
        single_curve_lengths = np.zeros(way_points.shape[1])
        way_points = self.calculate_length_in_mm(way_points)
        for i in range(way_points.shape[1]-1):
            single_curve_lengths[i+1] = np.linalg.norm(way_points[:,i]-way_points[:,i+1])
        total_curve_length = np.cumsum(single_curve_lengths)
        t = total_curve_length/(total_curve_length[-1])
        self.length_of_leaflet = total_curve_length[-1]
        print('self.length_of_leaflet: ', self.length_of_leaflet )
        

        return t
    
    def calculate_cubic_spline(self, way_points):
        t = self.calculate_spline_param(way_points)
       
        fx_cs = CubicSpline(t, way_points[0,:])
        fy_cs = CubicSpline(t, way_points[1,:])
        var = np.linspace(0, 1, num=100)
        x_cs = fx_cs(var)
        y_cs = fy_cs(var)
        return x_cs, y_cs


    def get_data_points(self):
        """Return all data points."""
        try:
            return self.data
        except KeyError:
            pass


    def add_new_spline_waypoint(self, new_waypoint, spline_index_new_waypoint):

        old_waypoints = self.get_waypoints()
        old_waypoints_reduced = old_waypoints[~np.all(old_waypoints == 0, axis=1)]
        spline_array = self.get_spline_array()
        new_spline_array = np.zeros((old_waypoints_reduced.shape[0]+1,2))
        distance =np.zeros(len(spline_array))

        spline_index_old_waypoint = np.zeros(old_waypoints_reduced.shape[0])

        for i in range(old_waypoints_reduced.shape[0]):
            distance_x = np.abs(spline_array[:,0]-old_waypoints_reduced[i,0])
            distance_y = np.abs(spline_array[:,1]-old_waypoints_reduced[i,1])
            distance = distance_x + distance_y
            spline_index_old_waypoint[i] = np.argmin(np.round(distance))

        index_new_waypoint = np.argmax(spline_index_old_waypoint>spline_index_new_waypoint)
        
        
        new_spline_array[0:index_new_waypoint,:] = old_waypoints_reduced[0:index_new_waypoint,:]
        new_spline_array[index_new_waypoint+1:new_spline_array.shape[0],:] = old_waypoints_reduced[index_new_waypoint:old_waypoints_reduced.shape[0],:]
        new_spline_array[index_new_waypoint,:] = new_waypoint

        
        
        return np.transpose(new_spline_array)
       
        
    def get_waypoints(self):
        return self.waypoints

    def set_waypoints(self, new_waypoints):
        self.waypoints = new_waypoints

    def get_spline_array(self):
        return self.data['pos']

    def get_spline_points(self):
        return self.data_points
    
    def set_spline_points_to_empty(self):
        self.data_points = []
        self.data['pos'] = []
        self.updateGraph()
    
    def add_circles_around_waypoints(self):

        waypoint_array = self.get_waypoints()
        waypoints_reduced = waypoint_array[~np.all(waypoint_array == 0, axis=1)]
        
        # Create a list of circle items
        circles = []  # list of QGraphicsEllipseItem objects
        pen = pg.mkPen('r')  # set the circle color

        for i in range(len(waypoints_reduced)):
            x, y = waypoints_reduced[i]
            radius = 1
            circle = pg.ScatterPlotItem(x=[x], y=[y], pen='b', brush=None, symbol='o', size=5*radius)
            circles.append(circle)

        # Add the circles to the plot
        for circle in circles:
            self.plot.addItem(circle)


    def draw_new_waypoint(self, new_waypoint):
        x, y = new_waypoint
        radius = 1
        circle = pg.ScatterPlotItem(x=[x], y=[y], pen='cyan', brush=None, symbol='o', size=5*radius)
        self.plot.addItem(circle)

                