import sys

import numpy as np
import random
import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import axes3d
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QKeySequence, QIntValidator
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout,
                               QHeaderView, QLabel, QMainWindow, QSlider,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget, QGridLayout, QRadioButton, QLineEdit, QFormLayout, QPushButton, QSizePolicy)

from mpr_slicer import MPR_Slicer
from model3D import Model_3D
from viewer3D import VtkDisplay

class Clipped_Spline_Length_Tab(QWidget):
    def __init__(self, mpr_slicer, tab_2D, excel_writer):
        super().__init__()

        self.mpr_slicer = mpr_slicer
        self.tab_2D = tab_2D
        self.excel_writer = excel_writer

        button_layout = QGridLayout()
        self.general_label = QLabel("Calculate the clipped leaflet length:")
        self.with_complete_model_button = QPushButton("With complete model")
        self.with_complete_model_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.with_complete_model_button.clicked.connect(self.with_complete_model_button_was_pressed)
        self.with_partial_model_button = QPushButton("With partial model")
        self.with_partial_model_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.with_partial_model_button.clicked.connect(self.with_partial_model_button_was_pressed)
        self.without_model_button = QPushButton("Without model")
        self.without_model_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.without_model_button.clicked.connect(self.without_model_button_was_pressed)
        self.neighbor_slice_button = QPushButton("With neighbor slice")
        self.neighbor_slice_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.neighbor_slice_button.clicked.connect(self.neighbor_slice_button_was_pressed)
        
        self.result_complete_model_label_ant = QLabel("")
        self.result_partial_model_label_ant = QLabel("")
        self.result_without_model_label_ant = QLabel("")
        self.result_neighbor_slice_label_ant = QLabel("")
        self.result_complete_model_label_post = QLabel("")
        self.result_partial_model_label_post = QLabel("")
        self.result_without_model_label_post = QLabel("")
        self.result_neighbor_slice_label_post = QLabel("")



        button_layout.addWidget(self.with_complete_model_button, 0, 0)
        button_layout.addWidget(self.with_partial_model_button, 0, 1)
        button_layout.addWidget(self.without_model_button, 0, 2)
        button_layout.addWidget(self.neighbor_slice_button, 0, 3)
        button_layout.addWidget(self.result_complete_model_label_ant,1,0)
        button_layout.addWidget(self.result_partial_model_label_ant,1,1)
        button_layout.addWidget(self.result_without_model_label_ant,1,2)
        button_layout.addWidget(self.result_neighbor_slice_label_ant,1,3)
        button_layout.addWidget(self.result_complete_model_label_post,2,0)
        button_layout.addWidget(self.result_partial_model_label_post,2,1)
        button_layout.addWidget(self.result_without_model_label_post,2,2)
        button_layout.addWidget(self.result_neighbor_slice_label_post,2,3)

        window_layout = QVBoxLayout()

        window_layout.addLayout(button_layout)
        self.setLayout(window_layout)

    def with_complete_model_button_was_pressed(self):
        clipped_length_complete_ant = self.tab_2D.length_complete_ant_total - self.tab_2D.length_complete_ant_visible 
        clipped_length_complete_post = self.tab_2D.length_complete_post_total - self.tab_2D.length_complete_post_visible

        self.result_complete_model_label_ant.setText('anterior:' + str(clipped_length_complete_ant) +' mm')
        self.result_complete_model_label_post.setText('posterior:' + str(clipped_length_complete_post) +' mm')
        self.excel_writer.add_value('pre', 'leaflet_length_clipped_complete_ant', clipped_length_complete_ant)
        self.excel_writer.add_value('pre', 'leaflet_length_clipped_complete_post', clipped_length_complete_post)
        self.excel_writer.add_value('intra', 'leaflet_length_clipped_complete_ant', clipped_length_complete_ant)
        self.excel_writer.add_value('intra', 'leaflet_length_clipped_complete_post', clipped_length_complete_post)

    
    def with_partial_model_button_was_pressed(self):
        clipped_length_partial_ant = self.tab_2D.length_partial_ant_total - self.tab_2D.length_partial_ant_visible 
        clipped_length_partial_post = self.tab_2D.length_partial_post_total - self.tab_2D.length_partial_post_visible

        self.result_partial_model_label_ant.setText('anterior:' + str(clipped_length_partial_ant) +' mm')
        self.result_partial_model_label_post.setText('posterior:' + str(clipped_length_partial_post) +' mm')
        self.excel_writer.add_value('pre', 'leaflet_length_clipped_partial_ant', clipped_length_partial_ant)
        self.excel_writer.add_value('pre', 'leaflet_length_clipped_partial_post', clipped_length_partial_post)
        self.excel_writer.add_value('intra', 'leaflet_length_clipped_partial_ant', clipped_length_partial_ant)
        self.excel_writer.add_value('intra', 'leaflet_length_clipped_partial_post', clipped_length_partial_post)

    def without_model_button_was_pressed(self):
        clipped_length_without_ant = self.tab_2D.length_without_ant_total - self.tab_2D.length_without_ant_visible 
        clipped_length_without_post = self.tab_2D.length_without_post_total - self.tab_2D.length_without_post_visible

        self.result_without_model_label_ant.setText('anterior:' + str(clipped_length_without_ant) +' mm')
        self.result_without_model_label_post.setText('posterior:' + str(clipped_length_without_post) +' mm')
        self.excel_writer.add_value('pre', 'leaflet_length_clipped_without_ant', clipped_length_without_ant)
        self.excel_writer.add_value('pre', 'leaflet_length_clipped_without_post', clipped_length_without_post)
        self.excel_writer.add_value('intra', 'leaflet_length_clipped_without_ant', clipped_length_without_ant)
        self.excel_writer.add_value('intra', 'leaflet_length_clipped_without_post', clipped_length_without_post)

    def neighbor_slice_button_was_pressed(self):
        clipped_length_neighbor_ant = self.tab_2D.length_neighbor_ant_total - self.tab_2D.length_neighbor_ant_visible 
        clipped_length_neighbor_post = self.tab_2D.length_neighbor_post_total - self.tab_2D.length_neighbor_post_visible

        self.result_neighbor_slice_label_ant.setText('anterior:' + str(clipped_length_neighbor_ant) +' mm')
        self.result_neighbor_slice_label_post.setText('posterior:' + str(clipped_length_neighbor_post) +' mm')
        self.excel_writer.add_value('pre', 'leaflet_length_clipped_neighbor_ant', clipped_length_neighbor_ant)
        self.excel_writer.add_value('pre', 'leaflet_length_clipped_neighbor_post', clipped_length_neighbor_post)
        self.excel_writer.add_value('intra', 'leaflet_length_clipped_neighbor_ant', clipped_length_neighbor_ant)
        self.excel_writer.add_value('intra', 'leaflet_length_clipped_neighbor_post', clipped_length_neighbor_post)