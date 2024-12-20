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
                               QWidget, QCheckBox, QGridLayout, QRadioButton, QLineEdit, QFormLayout, QPushButton, QSizePolicy,QDialog, QFileDialog)

from mpr_slicer import MPR_Slicer
from model3D import Model_3D
from viewer3D import VtkDisplay
import pathlib
import os



class Model_3D_Tab(QWidget):
    def __init__(self, mpr_slicer, volume_pre_op, volume_intra_op, mv_model_pre, ant_model_pre, post_model_pre, mv_model_intra, ant_model_intra, post_model_intra, slice_thickness_pre, slice_thickness_intra, apex_pre, apex_intra, excel_writer, dataloader):
        super().__init__()

        self.excel_writer = excel_writer
        self.dataloader = dataloader
    

        self.slicer_volume = volume_intra_op
        self.image_width = self.slicer_volume.shape[0] # x
        self.image_height = self.slicer_volume.shape[1] # y
        self.image_depth = self.slicer_volume.shape[2] #z
        self.max_extension = np.max([self.slicer_volume.shape[0], self.slicer_volume.shape[1], self.slicer_volume.shape[2]])
        self.slicer_extension = np.array([0, self.image_width, 0, self.image_height])

        self.mpr_slicer = mpr_slicer
        self.mpr_slicer.tab_3D = self
        self.mpr_slicer.mpr_width = self.slicer_volume.shape[0] # x
        self.mpr_slicer.mpr_height = self.slicer_volume.shape[1] # y
        self.mpr_slicer.mpr_depth = self.slicer_volume.shape[2] #z


        self.mpr_slicer.mv_model_intra = mv_model_intra
        self.mpr_slicer.ant_model_intra = ant_model_intra
        self.mpr_slicer.post_model_intra = post_model_intra
        self.mpr_slicer.mv_model_pre = mv_model_pre
        self.mpr_slicer.ant_model_pre = ant_model_pre
        self.mpr_slicer.post_model_pre = post_model_pre

        self.mpr_slicer.apex_pre = apex_pre
        self.mpr_slicer.apex_intra = apex_intra
        self.mpr_slicer.apex = apex_intra


        self.mpr_slicer.slice_thickness_pre = slice_thickness_pre
        self.mpr_slicer.slice_thickness_intra = slice_thickness_intra
        
        self.mpr_slicer.set_mv_model()


        self.mpr_plot = pg.PlotWidget()
        self.mpr_image = pg.ImageItem()
        colorMap = pg.colormap.get('gray', source = 'matplotlib')
        self.mpr_image.setColorMap(colorMap)
        self.mpr_image.setLevels([0, 255])


        self.mpr_image_array = self.mpr_slicer.get_mpr()
        self.mpr_image.setImage(self.mpr_image_array) 
        self.mpr_image.setLevels([0,255])

        self.mpr_plot.addItem(self.mpr_image)
        self.mpr_plot.getPlotItem().hideAxis("left")
        self.mpr_plot.getPlotItem().hideAxis("bottom")
        self.mpr_plot.getPlotItem().setMouseEnabled(x=False, y=False)


        self.mpr_plot.getPlotItem().setXRange(0, self.image_width) 
        self.mpr_plot.getPlotItem().setYRange(0, self.image_height) 
        self.mpr_plot.setFixedSize((self.image_width*3), (self.image_height*3))

        self.data_MPR = None
        self.data_Index = None
        self.data_World = None

        self.mpr_on_plane_shown = False
        self.show_mpr_set = True
        self.cropped_volume = False
        self.reverse_crop_direction = False
        self.hide_volume = False

        self.item_to_adjust_transparency = None
        self.item_to_adjust_visibility = None

        self.name_dict_plot_items = {'None': None, 'all': self.mpr_slicer.get_all, 'Volume': self.mpr_slicer.get_volume, 
                                     'croped volume': self.mpr_slicer.get_croped_volume,
                                     'MPR Mesh': self.mpr_slicer.get_mpr_plane_mesh, 
                                     'MPR Mesh helper points': self.mpr_slicer.get_mpr_mesh_helper_points, 
                                     'MPR Mesh coordinate system': self.mpr_slicer.get_mpr_coordinate_system,
                                     'annulus plane': self.mpr_slicer.get_annulus_plane_mesh,
                                     'annulus plane coordinate system': self.mpr_slicer.get_annulus_plane_coordinate_system, 
                                     'annulus plane center of gravity': self.mpr_slicer.get_annulus_plane_center_of_gravity,
                                     'mv mesh': self.mpr_slicer.get_mv_mesh, 'ant mesh': self.mpr_slicer.get_ant_mesh,
                                     'post mesh': self.mpr_slicer.get_post_mesh, 'apex': self.mpr_slicer.get_apex, 
                                     'World coordinate sytsem': self.mpr_slicer.get_world_coordi,
                                     'annulus': self.mpr_slicer.get_annulus,
                                     'intersection line': self.mpr_slicer.get_intersection_line,
                                     'coordinate system apex': self.mpr_slicer.get_coordinate_system_apex,
                                     'bounding box': self.mpr_slicer.get_bounding_box,
                                     'axis apex gravity center': self.mpr_slicer.get_axis_apex_gravity_center,
                                     'aortic point': self.mpr_slicer.get_aortic_point,
                                     'clip': self.mpr_slicer.get_clip}
        
        self.name_dict_plot_items_and_visibility = {'None': None, 'all': self.mpr_slicer.get_all_visibility, 'Volume': self.mpr_slicer.get_volume_and_visibility, 
                                     'croped volume': self.mpr_slicer.get_croped_volume_and_visibility,
                                     'MPR Mesh': self.mpr_slicer.get_mpr_plane_mesh_and_visibility, 
                                     'MPR Mesh helper points': self.mpr_slicer.get_mpr_mesh_helper_points_and_visibility, 
                                     'MPR Mesh coordinate system': self.mpr_slicer.get_mpr_coordinate_system_and_visibility,
                                     'annulus plane': self.mpr_slicer.get_annulus_plane_mesh_and_visibility,
                                     'annulus plane coordinate system': self.mpr_slicer.get_annulus_plane_coordinate_system_and_visibility, 
                                     'annulus plane center of gravity': self.mpr_slicer.get_annulus_plane_center_of_gravity_and_visibility,
                                     'mv mesh': self.mpr_slicer.get_mv_mesh_and_visibility, 'ant mesh': self.mpr_slicer.get_ant_mesh_and_visibility,
                                     'post mesh': self.mpr_slicer.get_post_mesh_and_visibility, 'apex': self.mpr_slicer.get_apex_and_visibility,
                                     'World coordinate sytsem': self.mpr_slicer.get_world_coordi_and_visibility, 
                                     'annulus': self.mpr_slicer.get_annulus_and_visibility,
                                     'intersection line': self.mpr_slicer.get_intersection_line_and_visibility,
                                     'coordinate system apex': self.mpr_slicer.get_coordinate_system_apex_and_visibility,
                                     'bounding box': self.mpr_slicer.get_bounding_box_and_visibility,
                                     'axis apex gravity center': self.mpr_slicer.get_axis_apex_gravity_center_and_visibility,
                                     'aortic point': self.mpr_slicer.get_aortic_point_and_visibility,
                                     'clip': self.mpr_slicer.get_clip_and_visibility}
        






 




        self.model_3D = Model_3D()


        grid = QGridLayout()
        grid.addWidget(self.mpr_plot, 0, 0)

        
        btn_layout=QGridLayout()
        
        slicer_type_layout=QGridLayout()
        
        checkBox = QCheckBox("Cut MPR automatically at box")
        checkBox.setChecked(True) 

        checkBox.stateChanged.connect(self.on_checkbox_state_changed)

        radio_button_layout = QHBoxLayout()
        self.normal_coordi_radio_button = QRadioButton("anatomical coordinate system with annulus plane normal")
        self.normal_coordi_radio_button.setChecked(True)
        self.normal_coordi_radio_button.toggled.connect(self.normal_coordi_selected)
        self.apex_coordi_radio_button = QRadioButton("anatomical coordinate system with apex axis")
        self.apex_coordi_radio_button.setChecked(False)
        self.apex_coordi_radio_button.toggled.connect(self.apex_coordi_selected)

        radio_button_layout.addWidget(self.normal_coordi_radio_button)
        radio_button_layout.addWidget(self.apex_coordi_radio_button)
        radio_button_layout.addWidget(checkBox)
        
        self.slice_set_number = QLineEdit()
        self.slice_set_number.setText(str('9'))
        self.slice_set_number.setValidator(QIntValidator())
        self.slice_set_number.setMaxLength(360)
        self.slice_set_number.setAlignment(Qt.AlignRight)
        self.slice_set_number_label = QLabel("slice_set_number: ")

        self.total_angel_set = QLineEdit()
        self.total_angel_set.setText(str('60'))
        self.total_angel_set.setValidator(QIntValidator())
        self.total_angel_set.setMaxLength(360)
        self.total_angel_set.setAlignment(Qt.AlignRight)
        self.total_angel_set_label = QLabel("total_angel_set: ")

        self.neighbor_angel = QLineEdit()
        self.neighbor_angel.setText(str('10'))
        self.neighbor_angel.setValidator(QIntValidator())
        self.neighbor_angel.setMaxLength(360)
        self.neighbor_angel.setAlignment(Qt.AlignRight)
        self.neighbor_angel_label = QLabel("neighbor_angel: ")
        
        self.p_0_x = QLineEdit()
        #self.p_0_x.setText(str('0'))
        self.p_0_x.setText(str('150'))
        self.p_0_x.setValidator(QIntValidator())
        self.p_0_x.setMaxLength(3)
        self.p_0_x.setAlignment(Qt.AlignRight)
        self.p_0_x_label = QLabel("p_0_x: ")

        self.p_0_y = QLineEdit()
        #self.p_0_y.setText(str('0'))
        self.p_0_y.setText(str('150'))
        self.p_0_y.setValidator(QIntValidator())
        self.p_0_y.setMaxLength(3)
        self.p_0_y.setAlignment(Qt.AlignRight)
        self.p_0_y_label = QLabel("p_0_y: ")

        self.p_0_z = QLineEdit()
        self.p_0_z.setText(str('0'))
        self.p_0_z.setValidator(QIntValidator())
        self.p_0_z.setMaxLength(3)
        self.p_0_z.setAlignment(Qt.AlignRight)
        self.p_0_z_label = QLabel("p_0_z: ")


        self.p_1_x = QLineEdit()
        #self.p_1_x.setText(str('0'))
        self.p_1_x.setText(str('150'))
        self.p_1_x.setValidator(QIntValidator())
        self.p_1_x.setMaxLength(3)
        self.p_1_x.setAlignment(Qt.AlignRight)
        self.p_1_x_label = QLabel("p_1_x: ")

        self.p_1_y = QLineEdit()
        #self.p_1_y.setText(str('0'))
        self.p_1_y.setText(str('150'))
        self.p_1_y.setValidator(QIntValidator())
        self.p_1_y.setMaxLength(3)
        self.p_1_y.setAlignment(Qt.AlignRight)
        self.p_1_y_label = QLabel("p_1_y: ")

        self.p_1_z = QLineEdit()
        #self.p_1_z.setText(str('0'))
        self.p_1_z.setText(str('300'))
        self.p_1_z.setValidator(QIntValidator())
        self.p_1_z.setMaxLength(3)
        self.p_1_z.setAlignment(Qt.AlignRight)
        self.p_1_z_label = QLabel("p_1_z: ")
        

        self.p_2_x = QLineEdit()
        #self.p_2_x.setText(str('0'))
        self.p_2_x.setText(str('10'))
        self.p_2_x.setValidator(QIntValidator())
        self.p_2_x.setMaxLength(3)
        self.p_2_x.setAlignment(Qt.AlignRight)
        self.p_2_x_label = QLabel("p_2_x: ")

        self.p_2_y = QLineEdit()
        #self.p_2_y.setText(str('0'))
        self.p_2_y.setText(str('10'))
        self.p_2_y.setValidator(QIntValidator())
        self.p_2_y.setMaxLength(3)
        self.p_2_y.setAlignment(Qt.AlignRight)
        self.p_2_y_label = QLabel("p_2_y: ")

        self.p_2_z = QLineEdit()
        #self.p_2_z.setText(str('0'))
        self.p_2_z.setText(str('10'))
        self.p_2_z.setValidator(QIntValidator())
        self.p_2_z.setMaxLength(3)
        self.p_2_z.setAlignment(Qt.AlignRight)
        self.p_2_z_label = QLabel("p_2_z: ")

        self.alpha = QLineEdit()
        self.alpha.setText(str('0'))
        self.alpha.setValidator(QIntValidator())
        self.alpha.setMaxLength(3)
        self.alpha.setAlignment(Qt.AlignRight)
        self.alpha_label = QLabel("alpha: ")

        self.beta = QLineEdit()
        self.beta.setText(str('0'))
        self.beta.setValidator(QIntValidator())
        self.beta.setMaxLength(3)
        self.beta.setAlignment(Qt.AlignRight)
        self.beta_label = QLabel("beta: ")

        self.gamma = QLineEdit()
        self.gamma.setText(str('0'))
        self.gamma.setValidator(QIntValidator())
        self.gamma.setMaxLength(3)
        self.gamma.setAlignment(Qt.AlignRight)
        self.gamma_label = QLabel("gamma: ")


        position_layout = QFormLayout()
        slicer_type_layout.addWidget(self.p_0_x_label, 0, 0)
        slicer_type_layout.addWidget(self.p_0_x, 0, 1)
        slicer_type_layout.addWidget(self.p_0_y_label, 0, 2) 
        slicer_type_layout.addWidget(self.p_0_y, 0, 3)
        slicer_type_layout.addWidget(self.p_0_z_label, 0, 4)       
        slicer_type_layout.addWidget(self.p_0_z, 0, 5)

        slicer_type_layout.addWidget(self.p_1_x_label, 1, 0)
        slicer_type_layout.addWidget(self.p_1_x, 1, 1)
        slicer_type_layout.addWidget(self.p_1_y_label, 1, 2) 
        slicer_type_layout.addWidget(self.p_1_y, 1, 3)
        slicer_type_layout.addWidget(self.p_1_z_label, 1, 4)       
        slicer_type_layout.addWidget(self.p_1_z, 1, 5)

        slicer_type_layout.addWidget(self.p_2_x_label, 2, 0)
        slicer_type_layout.addWidget(self.p_2_x, 2, 1)
        slicer_type_layout.addWidget(self.p_2_y_label, 2, 2) 
        slicer_type_layout.addWidget(self.p_2_y, 2, 3)
        slicer_type_layout.addWidget(self.p_2_z_label, 2, 4)       
        slicer_type_layout.addWidget(self.p_2_z, 2, 5)

        slicer_set_layout=QGridLayout()
        slicer_set_layout.addWidget(self.slice_set_number_label, 0, 0)
        slicer_set_layout.addWidget(self.slice_set_number, 0, 1)
        slicer_set_layout.addWidget(self.total_angel_set_label, 0, 2)
        slicer_set_layout.addWidget(self.total_angel_set, 0, 3)
        slicer_set_layout.addWidget(self.neighbor_angel_label,0,4)
        slicer_set_layout.addWidget(self.neighbor_angel,0,5)
        


        button_layout = QGridLayout()
        self.create_mpr_button = QPushButton("Create MPR")
        self.create_mpr_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.create_mpr_button.clicked.connect(self.create_mpr_button_was_pressed)
        self.create_mpr_through_clip_button = QPushButton("Create MPR through clip")
        self.create_mpr_through_clip_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.create_mpr_through_clip_button.clicked.connect(self.create_mpr_through_clip_button_was_pressed)
        #self.create_mpr_button.setEnabled(False)
        self.center_on_mpr_button = QPushButton("Center on MPR")
        self.center_on_mpr_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.center_on_mpr_button.clicked.connect(self.center_on_mpr_button_was_pressed)
        self.create_mpr_set_button = QPushButton("Create automatic MPR set")
        self.create_mpr_set_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.create_mpr_set_button.clicked.connect(self.create_mpr_set_button_was_pressed)
        self.show_mpr_set_button = QPushButton("Show automatic MPR set")
        self.show_mpr_set_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.show_mpr_set_button.clicked.connect(self.show_mpr_set_button_was_pressed)
        self.crop_volume_on_mpr_button = QPushButton("Crop volume on MPR")
        self.crop_volume_on_mpr_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.crop_volume_on_mpr_button.clicked.connect(self.crop_volume_on_mpr_button_was_pressed)
        self.reverse_crop_direction_button = QPushButton("Reverse crop direction")
        self.reverse_crop_direction_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.reverse_crop_direction_button.clicked.connect(self.reverse_crop_direction_button_was_pressed)
        self.hide_volume_button = QPushButton("Hide Volume")
        self.hide_volume_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.hide_volume_button.clicked.connect(self.hide_volume_button_was_pressed)
        self.save_single_mpr_button = QPushButton("Save single MPR")
        self.save_single_mpr_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.save_single_mpr_button.clicked.connect(self.save_single_mpr_button_was_pressed)
        self.save_mpr_set_button = QPushButton("Save MPR set")
        self.save_mpr_set_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.save_mpr_set_button.clicked.connect(self.save_mpr_set_button_was_pressed)
        self.test_bent_leaflet_button = QPushButton("Test bent leaflet")
        self.test_bent_leaflet_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.test_bent_leaflet_button.clicked.connect(self.test_bent_leaflet_button_was_pressed)
        self.change_volume_button = QPushButton("Change to pre op data")
        self.change_volume_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.change_volume_button.clicked.connect(self.change_volume_button_was_pressed)
        self.neighbor_mpr_button = QPushButton("Create neighbor MPR")
        self.neighbor_mpr_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.neighbor_mpr_button.clicked.connect(self.neighbor_mpr_button_was_pressed)
        self.change_model_button = QPushButton("Change mv model")
        self.change_model_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.change_model_button.clicked.connect(self.change_model_button_was_pressed)
        self.create_mpr_with_same_orientation_button = QPushButton("Create MPR with same orientation")
        self.create_mpr_with_same_orientation_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.create_mpr_with_same_orientation_button.clicked.connect(self.create_mpr_with_same_orientation_button_was_pressed)
        
        


        
        button_layout.addWidget(self.create_mpr_through_clip_button,0,0)
        button_layout.addWidget(self.create_mpr_with_same_orientation_button, 0,1)
        button_layout.addWidget(self.create_mpr_button, 0,2)
        button_layout.addWidget(self.create_mpr_set_button,0,3)
        button_layout.addWidget(self.neighbor_mpr_button,0,4)

        button_layout.addWidget(self.center_on_mpr_button,1,0)
        button_layout.addWidget(self.show_mpr_set_button,1,1)
        button_layout.addWidget(self.crop_volume_on_mpr_button,1,2)
        button_layout.addWidget(self.reverse_crop_direction_button,1,3)
        button_layout.addWidget(self.hide_volume_button,1,4)

        button_layout.addWidget(self.save_single_mpr_button,2,0)
        button_layout.addWidget(self.save_mpr_set_button,2,1)
        button_layout.addWidget(self.test_bent_leaflet_button,2,2)
        button_layout.addWidget(self.change_volume_button,2,3)
        button_layout.addWidget(self.change_model_button,2,4)
        
        

        self.slider_3d_x = QSlider(Qt.Horizontal)
        self.slider_3d_x.setMinimum(-180)
        self.slider_3d_x.setMaximum(180)
        self.slider_3d_x.setSingleStep(1)
        self.slider_3d_x.setValue(1)

        self.slider_3d_y = QSlider(Qt.Horizontal, self)
        self.slider_3d_y.setMinimum(-180)
        self.slider_3d_y.setMaximum(180)
        self.slider_3d_y.setSingleStep(1)
        self.slider_3d_y.setValue(0)
        
        self.slider_3d_z = QSlider(Qt.Horizontal, self)
        self.slider_3d_z.setMinimum(-180)
        self.slider_3d_z.setMaximum(180)
        self.slider_3d_z.setSingleStep(1)
        self.slider_3d_z.setValue(0)


        self.slider_plane_rotation = QSlider(Qt.Horizontal, self)
        self.slider_plane_rotation.setMinimum(0)
        self.slider_plane_rotation.setMaximum(180)
        self.slider_plane_rotation.setSingleStep(1)
        self.slider_plane_rotation.setValue(0)
        self.slider_plane_rotation_label = QLabel("Rotate the plane: ")
        
        self.slider_mpr_width = QSlider(Qt.Horizontal, self)
        self.slider_mpr_width.setMinimum(10)
        self.slider_mpr_width.setMaximum(self.max_extension)
        self.slider_mpr_width.setSingleStep(1)
        self.slider_mpr_width.setValue(self.image_width)
        self.slider_mpr_width_label = QLabel("MPR Width: ")

        self.slider_mpr_height = QSlider(Qt.Horizontal, self)
        self.slider_mpr_height.setMinimum(10)
        self.slider_mpr_height.setMaximum(self.max_extension)
        self.slider_mpr_height.setSingleStep(1)
        self.slider_mpr_height.setValue(self.image_height)
        self.slider_mpr_height_label = QLabel("MPR Height: ")


        self.slider_mpr_width.sliderMoved.connect(self.slider_mpr_width_was_moved)
        self.slider_mpr_height.sliderMoved.connect(self.slider_mpr_height_was_moved)
        

        slider_layout = QGridLayout()
        slider_layout.addWidget(self.slider_3d_x, 0, 0)
        slider_layout.addWidget(self.slider_3d_y, 0, 1)
        slider_layout.addWidget(self.slider_3d_z, 0, 2)

        mpr_size_layout = QGridLayout()
        mpr_size_layout.addWidget(self.slider_mpr_width_label, 0, 1)
        mpr_size_layout.addWidget(self.slider_mpr_width, 0, 2)
        mpr_size_layout.addWidget(self.slider_mpr_height_label, 0, 3)
        mpr_size_layout.addWidget(self.slider_mpr_height, 0, 4)

    

        axis_layout = QGridLayout()
        axis_layout.addWidget(self.slider_plane_rotation_label, 0, 1)
        axis_layout.addWidget(self.slider_plane_rotation, 0, 2)

        dropdown_layout = QGridLayout()

        #Dropdown
        self.label_transparency = QLabel('Adjust transparency:')
        self.dropdown_transparency = QComboBox()
        self.dropdown_transparency.addItem('None')
        self.dropdown_transparency.addItem('all')
        self.dropdown_transparency.addItem('Volume')
        self.dropdown_transparency.addItem('croped volume')
        self.dropdown_transparency.addItem('MPR Mesh')
        self.dropdown_transparency.addItem('MPR Mesh helper points')
        self.dropdown_transparency.addItem('MPR Mesh coordinate system')
        self.dropdown_transparency.addItem('annulus plane')
        self.dropdown_transparency.addItem('annulus plane coordinate system')
        self.dropdown_transparency.addItem('annulus plane center of gravity')
        self.dropdown_transparency.addItem('ant mesh')
        self.dropdown_transparency.addItem('post mesh')
        self.dropdown_transparency.addItem('mv mesh')
        self.dropdown_transparency.addItem('apex')
        self.dropdown_transparency.addItem('World coordinate sytsem')
        self.dropdown_transparency.addItem('annulus')
        self.dropdown_transparency.addItem('intersection line')
        self.dropdown_transparency.addItem('coordinate system apex')
        self.dropdown_transparency.addItem('bounding box')
        self.dropdown_transparency.addItem('axis apex gravity center')
        self.dropdown_transparency.addItem('aortic point')
        self.dropdown_transparency.addItem('clip')
        self.dropdown_transparency.currentIndexChanged.connect(self.item_transparency)

        self.slider_transparency = QSlider(Qt.Horizontal, self)
        self.slider_transparency.setMinimum(0)
        self.slider_transparency.setMaximum(10)
        self.slider_transparency.setSingleStep(1)
        self.slider_transparency.setValue(0)

        self.label_visibility = QLabel('Adjust visibility:')
        self.dropdown_visibility = QComboBox()
        self.dropdown_visibility.addItem('None')
        self.dropdown_visibility.addItem('all')
        self.dropdown_visibility.addItem('Volume')
        self.dropdown_visibility.addItem('croped volume')
        self.dropdown_visibility.addItem('MPR Mesh')
        self.dropdown_visibility.addItem('MPR Mesh helper points')
        self.dropdown_visibility.addItem('MPR Mesh coordinate system')
        self.dropdown_visibility.addItem('annulus plane')
        self.dropdown_visibility.addItem('annulus plane coordinate system')
        self.dropdown_visibility.addItem('annulus plane center of gravity')
        self.dropdown_visibility.addItem('ant mesh')
        self.dropdown_visibility.addItem('post mesh')
        self.dropdown_visibility.addItem('mv mesh')
        self.dropdown_visibility.addItem('apex')
        self.dropdown_visibility.addItem('World coordinate sytsem')
        self.dropdown_visibility.addItem('annulus')
        self.dropdown_visibility.addItem('intersection line')
        self.dropdown_visibility.addItem('coordinate system apex')
        self.dropdown_visibility.addItem('bounding box')
        self.dropdown_visibility.addItem('axis apex gravity center')
        self.dropdown_visibility.addItem('aortic point')
        self.dropdown_visibility.addItem('clip')
        self.dropdown_visibility.currentIndexChanged.connect(self.item_visibility)

        self.slider_visibility = QSlider(Qt.Horizontal, self)
        self.slider_visibility.setMinimum(0)
        self.slider_visibility.setMaximum(1)
        self.slider_visibility.setSingleStep(1)
        self.slider_visibility.setValue(1)

        self.slider_transparency.sliderMoved.connect(self.slider_transparency_was_moved)
        self.slider_visibility.sliderMoved.connect(self.slider_visibility_was_moved)

        dropdown_layout.addWidget(self.label_transparency, 0, 0)
        dropdown_layout.addWidget(self.dropdown_transparency, 0, 1)
        dropdown_layout.addWidget(self.slider_transparency, 0, 2)
        dropdown_layout.addWidget(self.label_visibility, 1, 0)
        dropdown_layout.addWidget(self.dropdown_visibility, 1, 1)
        dropdown_layout.addWidget(self.slider_visibility, 1, 2)




        window_layout = QVBoxLayout()
        window_layout.addLayout(grid)
        window_layout.addLayout(radio_button_layout)
        window_layout.addLayout(btn_layout)
        window_layout.addLayout(slider_layout)
        window_layout.addLayout(mpr_size_layout)
        window_layout.addLayout(slicer_set_layout)
        window_layout.addLayout(slicer_type_layout)
        window_layout.addLayout(position_layout)
        window_layout.addLayout(axis_layout)
        window_layout.addLayout(button_layout)
        window_layout.addLayout(dropdown_layout)
        self.setLayout(window_layout)

        
        
    def on_checkbox_state_changed(self, state):
        if state == 2:
            self.mpr_slicer.cut_mpr_auto =True
        elif state == 0:
            self.mpr_slicer.cut_mpr_auto =False


    def create_mpr_button_was_pressed(self):
        position_points = self.update_position_points()

        self.mpr_slicer.calculate_mpr(self.slicer_extension, position_points, True)
    
        
    def create_mpr_through_clip_button_was_pressed(self):
        clip_position = self.mpr_slicer.get_clip_mpr_position()
        clip_position = self.mpr_slicer.extend_clip_position(clip_position)
        self.mpr_slicer.position = clip_position
        self.mpr_slicer.clip_position_intra = clip_position
        self.mpr_slicer.save_clip_position()
        self.mpr_slicer.calculate_mpr(self.slicer_extension, clip_position, True)
        self.mpr_slicer.mpr_through_clip = self.mpr_slicer.mpr
    
    

    def center_on_mpr_button_was_pressed(self):
        mpr_center = self.mpr_slicer.get_mpr_center()
        mpr_normal = self.mpr_slicer.get_mpr_normal()
        distance_camera_mpr = 100
        new_camera_position = mpr_center + mpr_normal * distance_camera_mpr


    def create_mpr_set_button_was_pressed(self):
        slice_number = int(self.slice_set_number.text())
        total_angle = int(self.total_angel_set.text())
        self.mpr_slicer.get_automatic_slides(slice_number, total_angle)


    def show_mpr_set_button_was_pressed(self):
        self.show_mpr_set = not self.show_mpr_set
        if self.show_mpr_set == False:
            self.show_mpr_set_button.setText('Show MPR set')
            self.mpr_slicer.remove_all_slices()

        if self.show_mpr_set == True:
            self.show_mpr_set_button.setText('Hide MPR set')
            self.mpr_slicer.plot_all_slices()

    
    def crop_volume_on_mpr_button_was_pressed(self):
        self.cropped_volume = not self.cropped_volume
        if self.cropped_volume == True:
            self.crop_volume_on_mpr_button.setText('Show entire Volume')
            self.mpr_slicer.crop_along_plane(self.reverse_crop_direction)
            
        
        if self.cropped_volume == False:
            self.crop_volume_on_mpr_button.setText('Crop volume on MPR')
            self.mpr_slicer.show_entire_volume()


    def reverse_crop_direction_button_was_pressed(self):
        self.reverse_crop_direction = not self.reverse_crop_direction
        if self.cropped_volume == True:
            self.mpr_slicer.crop_along_plane(self.reverse_crop_direction)


    def hide_volume_button_was_pressed(self):
        self.hide_volume = not self.hide_volume
        if self.hide_volume == True:
            self.hide_volume_button.setText('Show volume')
            self.mpr_slicer.hide_volume()
        if self.hide_volume == False:
            self.hide_volume_button.setText('Hide volume')
            self.mpr_slicer.show_volume()


    def save_single_mpr_button_was_pressed(self):
        self.mpr_slicer.save_single_mpr()


    def save_mpr_set_button_was_pressed(self):
        self.mpr_slicer.save_mpr_set()


    def test_bent_leaflet_button_was_pressed(self):
        neighbor_angle = int(self.neighbor_angel.text())
        self.mpr_slicer.test_bent_leaflet(neighbor_angle)

    
    def neighbor_mpr_button_was_pressed(self):
        #angle in excel
        neighbor_angle = int(self.neighbor_angel.text())
        self.mpr_slicer.get_neighbor_mpr(neighbor_angle)


    def change_volume_button_was_pressed(self):
        self.mpr_slicer.change_volume_mode()
        self.mpr_slicer.plot.remove(self.mpr_slicer.plot.get_actors(), self.mpr_slicer.plot.get_meshes())
        self.mpr_slicer.remove_rest()
        self.mpr_slicer.set_volume()
        self.mpr_slicer.set_mv_model()
        self.mpr_slicer.define_anatomic_coordi(self.mpr_slicer.annulus_points)
        self.mpr_slicer.plot.remove()
        
        if self.mpr_slicer.mpr_mode_pre == True:
            self.change_volume_button.setText('Change to intra op data')
        elif self.mpr_slicer.mpr_mode_pre == False:
            self.change_volume_button.setText('Change to pre op data')


        self.mpr_slicer.prepare_visualization(True)


    def change_model_button_was_pressed(self):
        # Create an instance of MyPopup
        self.popup = MyPopup(self)
        # Set the popup to be a modal dialog (optional)
        self.popup.setWindowModality(Qt.ApplicationModal)
        # Show the popup
        self.popup.show()


    def create_mpr_with_same_orientation_button_was_pressed(self):
        position_points = self.mpr_slicer.reuse_mpr_orientation()
        self.mpr_slicer.calculate_percent_AP() 
        self.mpr_slicer.calculate_mpr(self.slicer_extension, position_points, True)


    def item_transparency(self, index):
        self.item_to_adjust_transparency = self.dropdown_transparency.currentText()

        
    def item_visibility(self, index):
        self.item_to_adjust_visibility = self.dropdown_visibility.currentText()


    def update_mpr(self, pre, bent):
        if pre == True and bent == False:
            mpr = self.mpr_slicer.get_pre_mpr()
        elif pre == False and bent == False:
            mpr = self.mpr_slicer.get_intra_mpr()
        elif pre == True and bent == True:
            mpr = self.mpr_slicer.get_bent_pre_mpr()
        elif pre == False and bent == True:
            mpr = self.mpr_slicer.get_bent_intra_mpr()
        self.mpr_image.setImage(mpr)
        self.mpr_image.setLevels([0,255])
            

    def update_position_points(self):
        self.position_axis = np.array([[int(self.p_0_x.text()), int(self.p_0_y.text()), int(self.p_0_z.text())],
                                        [int(self.p_1_x.text()), int(self.p_1_y.text()), int(self.p_1_z.text())],
                                        [int(self.p_2_x.text()), int(self.p_2_y.text()), int(self.p_2_z.text())]])
        position_points = self.position_axis
        return position_points

    def slider_mpr_width_was_moved(self):
        self.slicer_extension[1] = self.slider_mpr_width.value()


    def slider_mpr_height_was_moved(self):
        self.slicer_extension[3] = self.slider_mpr_height.value()


    def slider_transparency_was_moved(self):
        item_in_plot = self.name_dict_plot_items[self.item_to_adjust_transparency]()
        if item_in_plot is None:
            pass
        else:
            if self.item_to_adjust_transparency == 'Volume' or self.item_to_adjust_transparency == 'croped volume':
                volume_TF = True
            else:
                volume_TF = False
            self.mpr_slicer.adjust_transparency(item_in_plot, (self.slider_transparency.value()/10), volume_TF)


    def slider_visibility_was_moved(self):
        item_in_plot, visibility = self.name_dict_plot_items_and_visibility[self.item_to_adjust_visibility]()
        if item_in_plot is None:
            pass
        else:
            self.mpr_slicer.adjust_visibility(item_in_plot, visibility)



    def update_slicer_position_gui(self):
        if self.slicer_type == 'angles':
            self.p_1_x_label.setVisible(False)
            self.p_1_y_label.setVisible(False)
            self.p_1_z_label.setVisible(False)

            self.p_1_x.setVisible(False)
            self.p_1_y.setVisible(False)
            self.p_1_z.setVisible(False)

            self.p_2_x_label.setVisible(False)
            self.p_2_y_label.setVisible(False)
            self.p_2_z_label.setVisible(False)

            self.p_2_x.setVisible(False)
            self.p_2_y.setVisible(False)
            self.p_2_z.setVisible(False)

            self.alpha_label.setVisible(True)
            self.beta_label.setVisible(True)
            self.gamma_label.setVisible(True)

            self.alpha.setVisible(True)
            self.beta.setVisible(True)
            self.gamma.setVisible(True)


            self.slider_plane_rotation_label.setVisible(False)
            self.slider_plane_rotation.setVisible(False)
            
        elif self.slicer_type == 'axis':
            self.p_1_x_label.setVisible(True)
            self.p_1_y_label.setVisible(True)
            self.p_1_z_label.setVisible(True)

            self.p_1_x.setVisible(True)
            self.p_1_y.setVisible(True)
            self.p_1_z.setVisible(True)

            self.p_2_x_label.setVisible(True)
            self.p_2_y_label.setVisible(True)
            self.p_2_z_label.setVisible(True)

            self.p_2_x.setVisible(True)
            self.p_2_y.setVisible(True)
            self.p_2_z.setVisible(True)

            self.alpha_label.setVisible(False)
            self.beta_label.setVisible(False)
            self.gamma_label.setVisible(False)

            self.alpha.setVisible(False)
            self.beta.setVisible(False)
            self.gamma.setVisible(False)


            self.slider_plane_rotation_label.setVisible(True)
            self.slider_plane_rotation.setVisible(True)

    def normal_coordi_selected(self):
        self.mpr_slicer.normal_coordi = True
        self.mpr_slicer.calculate_best_fit_plane_of_annulus()


    def apex_coordi_selected(self):
        if self.mpr_slicer.apex is None:
            print('The given 3D has no specified apex point. The coordinate system will be calculated with the annulus normal.')
       
        else:
            self.mpr_slicer.normal_coordi = False
            self.mpr_slicer.calculate_best_fit_plane_of_annulus()
            

           

class MyPopup(QDialog):
    def __init__(self, my_gui):
        QDialog.__init__(self)
        self.my_gui = my_gui
        self.excel_writer = self.my_gui.excel_writer
        self.mpr_slicer = self.my_gui.mpr_slicer
        self.dataloader = my_gui.dataloader
        self.valve_model = 'complete'
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

                            # User,pre:mv,   ant,  post,   raw,   dat, spline,post:mv,  ant,  post,  raw, dat, spline
        self.filled_fields= [False, False, False, False, False, False, False, False, False, False, False, False, False]


        
        radio_button_layout = QHBoxLayout()
        self.complete_model_radio_button = QRadioButton("complete model")
        self.complete_model_radio_button.setChecked(True)
        self.complete_model_radio_button.toggled.connect(self.complete_model_selected)
        self.partial_model_radio_button = QRadioButton("partial model")
        self.partial_model_radio_button.setChecked(False)
        self.partial_model_radio_button.toggled.connect(self.partial_model_selected)

        radio_button_layout.addWidget(self.complete_model_radio_button)
        radio_button_layout.addWidget(self.partial_model_radio_button)


        self.file_lab_mv_pre = QLabel("Choose pre mv valve:") 
        self.file_text_mv_pre = QLineEdit(readOnly=True)
        self.file_button_mv_pre = QPushButton("Select file")
        self.file_lab_ant_pre = QLabel("Choose pre ant leaflet:") 
        self.file_text_ant_pre = QLineEdit(readOnly=True)
        self.file_button_ant_pre = QPushButton("Select file")
        self.file_lab_post_pre = QLabel("Choose pre post leaflet:") 
        self.file_text_post_pre = QLineEdit(readOnly=True)
        self.file_button_post_pre = QPushButton("Select file")


        self.file_lab_mv_intra = QLabel("Choose intra mv valve:") 
        self.file_text_mv_intra = QLineEdit(readOnly=True)
        self.file_button_mv_intra = QPushButton("Select file")
        self.file_lab_ant_intra = QLabel("Choose intra ant leaflet:") 
        self.file_text_ant_intra = QLineEdit(readOnly=True)
        self.file_button_ant_intra = QPushButton("Select file")
        self.file_lab_post_intra = QLabel("Choose intra post leaflet:") 
        self.file_text_post_intra = QLineEdit(readOnly=True)
        self.file_button_post_intra = QPushButton("Select file")

        self.file_lab_spline_model_pre = QLabel("Choose spline model pre file:") 
        self.file_text_spline_model_pre = QLineEdit(readOnly=True)
        self.file_button_spline_model_pre = QPushButton("Select file")
        self.file_lab_spline_model_intra = QLabel("Choose spline model intra file:") 
        self.file_text_spline_model_intra = QLineEdit(readOnly=True)
        self.file_button_spline_model_intra = QPushButton("Select file")

        self.file_button_mv_pre.clicked.connect(lambda: self.set_filepath('mv', 'pre', self.valve_model))
        self.file_button_ant_pre.clicked.connect(lambda: self.set_filepath('ant', 'pre', self.valve_model))
        self.file_button_post_pre.clicked.connect(lambda: self.set_filepath( 'post', 'pre', self.valve_model))

        self.file_button_mv_intra.clicked.connect(lambda: self.set_filepath('mv', 'intra', self.valve_model))
        self.file_button_ant_intra.clicked.connect(lambda: self.set_filepath('ant', 'intra', self.valve_model))
        self.file_button_post_intra.clicked.connect(lambda: self.set_filepath( 'post', 'intra', self.valve_model))

        self.file_button_spline_model_pre.clicked.connect(lambda: self.set_filepath('spline', 'pre', self.valve_model))
        self.file_button_spline_model_intra.clicked.connect(lambda: self.set_filepath('spline', 'intra', self.valve_model))

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

        load_layout.addWidget(self.file_lab_mv_intra , 0, 3)
        load_layout.addWidget(self.file_text_mv_intra , 0, 4)
        load_layout.addWidget(self.file_button_mv_intra , 0, 5)
        load_layout.addWidget(self.file_lab_ant_intra, 1, 3)
        load_layout.addWidget(self.file_text_ant_intra,1, 4)
        load_layout.addWidget(self.file_button_ant_intra, 1, 5)
        load_layout.addWidget(self.file_lab_post_intra,2, 3)
        load_layout.addWidget(self.file_text_post_intra,2, 4)
        load_layout.addWidget(self.file_button_post_intra,2, 5)

        load_layout.addWidget(self.file_lab_spline_model_pre, 5, 0)
        load_layout.addWidget(self.file_text_spline_model_pre,5, 1)
        load_layout.addWidget(self.file_button_spline_model_pre, 5, 2)
        load_layout.addWidget(self.file_lab_spline_model_intra,5, 3)
        load_layout.addWidget(self.file_text_spline_model_intra,5, 4)
        load_layout.addWidget(self.file_button_spline_model_intra,5, 5)

        start_button_layout = QHBoxLayout()
        self.start_button = QPushButton("Load new model")
        self.start_button.setEnabled(False)
        start_button_layout.addWidget(self.start_button,0)

        self.start_button.clicked.connect(self.start_button_was_pressed)

        window_layout = QGridLayout()
        window_layout.addLayout(radio_button_layout, 1, 0)
        window_layout.addLayout(load_layout, 2,0)
        window_layout.addLayout(start_button_layout, 3,0)
        
        self.setLayout(window_layout)
    
    def start_button_was_pressed(self):
        all_files = self.dataloader.prepare_all_files()
        self.mpr_slicer.unpack_all_files(all_files)
        self.mpr_slicer.prepare_model()
        self.close()


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
            
        elif mode == 'spline' and pre_or_intra == 'pre':
            file_text = self.file_text_spline_model_pre
            filter = "MAT files (*.mat)"      
        elif mode == 'spline' and pre_or_intra == 'intra':
            file_text = self.file_text_spline_model_intra
            filter = "MAT files (*.mat)"                                                        #set the filter to just show h5 data

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

        if np.all(self.filled_fields) == True:
            self.start_button.setEnabled(True)
        return
    
    def cut_mpr_automatically(self):
        self.mpr_slicer.cut_mpr_at_bounding_box == True
    
    def complete_model_selected(self):
        self.valve_model = 'complete'


    def partial_model_selected(self):
        self.valve_model = 'partial'
