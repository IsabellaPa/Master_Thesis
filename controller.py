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
from PySide6.QtGui import QAction, QPainter,QImage, QPixmap, QPen, QIntValidator, QColor, QKeySequence, QPolygon, QPolygonF, QBrush, QMouseEvent, QFont, QCursor
from PySide6.QtWidgets import (QApplication, QLabel,QMainWindow, QPushButton, QWidget, QListWidget,
                               QLineEdit,QFileDialog,QVBoxLayout, QHBoxLayout,QDialogButtonBox,
                               QGridLayout, QCheckBox, QMessageBox, QSizePolicy, QSlider, QGraphicsView, 
                               QGraphicsScene, QStyle,QSpacerItem, QRadioButton)
from random import randint, choice
import matplotlib.patches as mpatches
import pyqtgraph as pg
from scipy.interpolate import CubicSpline, UnivariateSpline, LSQUnivariateSpline

from animation import Animation
from dataloader import Dataloader
from leaflet_graph import Leaflet_Graph
from gui import Gui



class Controller():
    def __init__(self, master=None):
        self.my_dataloader = Dataloader()
        self.dicom_information = self.my_dataloader.get_data_from_path()
        self.data = self.my_dataloader.read_image_data()
        self.my_animation = Animation(self.data, self.dicom_information)
        self.my_gui = Gui(self.my_animation, self.my_dataloader)
 

if __name__ == "__main__":

    app = QApplication(sys.argv)
    w = Controller()
    sys.exit(app.exec())