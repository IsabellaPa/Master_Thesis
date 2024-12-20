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
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkIOLegacy import vtkStructuredPointsReader
import re
import scipy.io
from vedo import *
import ast


class Dataloader():
    def __init__(self):
        self.path  = #
        #only the image data
        self.data = None
        #the whole dicom information
        self.dicom_information  = None
        self.prepare_nr_of_phase_to_model()


        self.mv_model_pre_path = None
        self.ant_model_pre_path = None
        self.post_model_pre_path = None
        self.spline_model_pre_path = None
        self.dat_data_pre_path = None
        self.raw_data_pre_path = None
        self.apex_clip_pre_path = None
        self.annulus_pre_path = None

        self.mv_model_intra_path = None
        self.ant_model_intra_path = None
        self.post_model_intra_path = None
        self.spline_model_intra_path = None
        self.dat_data_intra_path = None
        self.raw_data_intra_path = None
        self.apex_intra = None
        self.apex_clip_intra_path = None
        self.annulus_intra_path = None



    def prepare_nr_of_phase_to_model(self):
        self.number_of_phase_to_volume = {
            'mv0': 0,
            'mv1': 1,
            'mv2': 2,
            'mv3': 3,
            'mv4': 4,
            'mv5': 5,
            'mv6': 6,
            'mv7': 7,
            'mv8': 8
            
        }

    def prepare_all_files(self):
        try:
            volume_phase_pre = self.number_of_phase_to_volume[os.path.splitext(os.path.basename(self.mv_model_pre_path))[0]]
        except:
            volume_phase_pre = 0
        try:
            volume_phase_intra = self.number_of_phase_to_volume[os.path.splitext(os.path.basename(self.mv_model_intra_path))[0]]
        except:
            volume_phase_intra = 0

        self.volume_pre_op, self.dim_size_pre = self.open_volume(self.raw_data_pre_path, self.dat_data_pre_path, volume_phase_pre)
        self.volume_intra_op, self.dim_size_intra = self.open_volume(self.raw_data_intra_path, self.dat_data_intra_path, volume_phase_intra)

        
        
        self.model_mv_pre, self.model_ant_pre, self.model_post_pre = self.load_valve_models(self.mv_model_pre_path, self.ant_model_pre_path, self.post_model_pre_path)
        
        
        self.model_mv_intra, self.model_ant_intra, self.model_post_intra = self.load_valve_models(self.mv_model_intra_path, self.ant_model_intra_path, self.post_model_intra_path)
         
        
        self.slice_thickness_pre = self.get_volume_slice_thickness(self.dat_data_pre_path)# get_slice_thickness()
        self.slice_thickness_intra = self.get_volume_slice_thickness(self.dat_data_intra_path)

        if self.mv_model_pre_path is not None and self.spline_model_pre_path is not None:
            self.spline_model_pre = self.load_spline_model(self.mv_model_pre_path, self.spline_model_pre_path)
        else:
            self.spline_model_pre =None
        if self.mv_model_intra_path is not None and self.spline_model_intra_path is not None:
            self.spline_model_intra = self.load_spline_model(self.mv_model_intra_path, self.spline_model_intra_path)
        else:
            self.spline_model_intra = None

        self.apex_pre, self.clip_start_pre, self.clip_end_pre = self.read_apex_clip_file(self.apex_clip_pre_path)
        self.apex_intra, self.clip_start_intra, self.clip_end_intra = self.read_apex_clip_file(self.apex_clip_intra_path)

        if self.annulus_pre_path is not None:
            self.annulus_pre_data = self.read_annulus_file(self.annulus_pre_path)
        else:
            self.annulus_pre_data = None

        if self.annulus_intra_path is not None:
            self.annulus_intra_data = self.read_annulus_file(self.annulus_intra_path)
        else:
            self.annulus_intra_data = None
        return [self.volume_pre_op, self.volume_intra_op, self.model_mv_pre, self.model_ant_pre, self.model_post_pre, self.model_mv_intra, self.model_ant_intra, self.model_post_intra, self.slice_thickness_pre, self.slice_thickness_intra, self.spline_model_pre, self.spline_model_intra, self.apex_pre, self.apex_intra, None, self.clip_start_intra, None, self.clip_end_intra, self.annulus_pre_data, self.annulus_intra_data]
        
    def read_image_data(self):
        self.data = self.dicom_information.pixel_array
        self.data = self.data
        return self.data
    
    def print_dicom_information(self):
        print(self.dicom_information)

    def get_data_from_path(self):
        self.dicom_information = dcmread(self.path)
        return self.dicom_information
    

    def load_example_vtk_data(self):
        self.data =vtkSphereSource()
        return self.data
    
    
    def open_volume(self, raw_file, dat_file, needed_phase):
        Dim_size = self.get_volume_dimensions(dat_file)

        nr_of_phases = 0
        with open(dat_file, 'r') as file:
            for line in file:
                match = re.search(r'NbrPhases:\\s+(\\d+)', line)
                if not match:
                    match = re.search(r'NbrPhases:\x20*(\d+)', line)
                
                if match:
                    nr_of_phases = int(match.group(1))
        
        if nr_of_phases == 0:

            f = open(raw_file,'rb') #only opens the file for reading
            img_arr=np.fromfile(f,dtype=np.uint8)

            img_arr = img_arr[0:Dim_size[0]*Dim_size[1]*Dim_size[2]] 
            img_arr = img_arr.reshape(Dim_size[2],Dim_size[1],Dim_size[0]) 
            img_arr_slice = img_arr[:, :, 20 ]
        
        else: 
            with open(raw_file, 'rb') as f:
                raw_data = np.fromfile(f, dtype=np.uint8)

            # Reshape the data into a 3D array (images, height, width)
            raw_data = raw_data.reshape(nr_of_phases, Dim_size[2],Dim_size[1],Dim_size[0])

            img_arr = raw_data[needed_phase, :, :]
            
            
    
        '''plt_py.imshow(img_arr_slice)
        plt_py.show()'''

        volume =np.transpose(img_arr)
        return volume, Dim_size
    
    def get_volume_dimensions(self, file_path):

        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(r'Resolution:\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)', line)
                if not match:
                    match = re.search(r'Resolution:\x20*(\d+)\x20*(\d+)\x20*(\d+)', line)
                if match:
                    dimensions = [int(match.group(i)) for i in range(1, 4)]
        return dimensions

    def get_volume_slice_thickness(self, file_path):

            with open(file_path, 'r') as file:
                for line in file:
                    match = re.search(r'SliceThickness:\\s+(\\d+(\\.\\d+)?)\\s+(\\d+(\\.\\d+)?)\\s+(\\d+(\\.\\d+)?)', line)
                    if not match:
                        match = re.search(r'SliceThickness:\x20*(\d+(\.\d+)?)\x20*(\d+(\.\d+)?)\x20*(\d+(\.\d+)?)', line)
                    if match:
                        if '.' in match.group():
                            thickness = [float(match.group(i)) for i in range(1, 4)]
                        else:
                                match = re.search(r'SliceThickness:\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)', line)
                                thickness = [int(match.group(i)) for i in range(1, 4)]

            return thickness
    
    def load_inp_file(self, file_path):
        vertices = []
        faces = []

        try:
            with open(file_path, 'r') as file:
                # Read the first line to get the number of vertices and faces
                num_vertices, num_faces, _, _, _ = map(int, file.readline().split())

                # Read vertices
                for _ in range(num_vertices):
                    line = file.readline().split()
                    vertice_id, x, y, z = int(line[0]), float(line[1]), float(line[2]), float(line[3])
                    vertices.append((x, y, z))

                # Read faces
                for _ in range(num_faces):
                    line = file.readline().split()
                    face_id, _, face_type, v1, v2, v3 = int(line[0]), int(line[1]), line[2], int(line[3]), int(line[4]), int(line[5])
                    faces.append((v1, v2, v3))
                    

        except FileNotFoundError:
            print(f"File not found: {file_path}")

        except Exception as e:
            print(f"An error occurred: {e}")

        return  vertices, faces
    
    def load_valve_models(self, model_mv_path, model_ant_path, model_post_path):
        
        if model_mv_path is not None:
            model_mv_vertices, model_mv_faces = self.load_inp_file(model_mv_path)
            model_mv = [model_mv_vertices, model_mv_faces]
        else:
            model_mv = None

        if model_ant_path is not None:
            model_ant_vertices, model_ant_faces = self.load_inp_file(model_ant_path)
            model_ant = [model_ant_vertices, model_ant_faces]
        else:
            model_ant = None

        if model_post_path is not None:
            model_post_vertices, model_post_faces = self.load_inp_file(model_post_path)
            model_post = [model_post_vertices, model_post_faces]
        else:
            model_post = None

        if model_post is not None and model_ant is not None:
            new_ant_v, new_ant_f, new_post_v, new_post_f = self.split_ant_and_post(model_ant, model_post)

            model_ant = [new_ant_v, new_ant_f]
            model_post = [new_post_v, new_post_f]
        

        return model_mv, model_ant, model_post
    
    def read_apex_clip_file(self, apex_clip_file):
        with open(apex_clip_file, 'r') as file:
            content = file.read()
            apex, clip_start, clip_end = np.zeros(3)
            for line in content.splitlines():
                if line.startswith('apex: '):
                    apex = np.array(ast.literal_eval(line.split(': ')[1]))
                elif line.startswith('clip_start: '):
                    clip_start = np.array(ast.literal_eval(line.split(': ')[1]))
                elif line.startswith('clip_end: '):
                    clip_end = np.array(ast.literal_eval(line.split(': ')[1]))
        return apex, clip_start, clip_end

    def read_annulus_file(self, file_path):
        pass

    
    
    
    def split_ant_and_post(self, model_1, model_2):
        edges_model_1 = model_1[1]
        edges_model_2 = model_2[1]

        index_used_vertices_model_1 = np.unique(edges_model_1)
        index_used_vertices_model_2 = np.unique(edges_model_2)

        vertices_model_1= np.array(model_1[0])
        vertices_model_2= np.array(model_2[0])

        used_vertices_ant = vertices_model_1[index_used_vertices_model_1]
        used_vertices_post = vertices_model_2[index_used_vertices_model_2]

        index_mapping_ant = {value: index for index, value in enumerate(index_used_vertices_model_1)}
        new_id_ant = np.vectorize(index_mapping_ant.get)(edges_model_1)

        index_mapping_post = {value: index for index, value in enumerate(index_used_vertices_model_2)}
        new_id_post = np.vectorize(index_mapping_post.get)(edges_model_2)
        
        return used_vertices_ant, new_id_ant, used_vertices_post, new_id_post

                
    def get_point_from_inp_file(self, file_path_model, point):
        pattern = re.compile('^'+ str(point) + r'\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)')

        with open(file_path_model, 'r') as file:
            for line in file:
                matches = pattern.findall(line)
                for match in matches:
                    nr_x_values = [float(num) for num in match]
                    return nr_x_values
                
    def load_spline_model(self, path_inp, path_mat):
        mat = scipy.io.loadmat(path_mat)
        param_points = mat['aws']
        all_points_of_spline = np.zeros((79,39,3))
        for i in range(0, 79): 
            param_points = mat['aws'][:,i]
            points_of_spline = np.zeros((param_points.shape[0]-1, 3))
            for j in range(0, param_points.shape[0]-1):
                point= param_points[j]
                new_point = self.get_point_from_inp_file(path_inp, point)
                points_of_spline[j, :] = new_point
            all_points_of_spline[i,:,:]= points_of_spline
        return all_points_of_spline
    
    
    def load_u_v_model(self):
        model_mv_path = r'C:\\Users\\ipatzke\\OneDrive - Philips\\Documents\\Masterarbeit\\MasterMission\\Resourcen\\Tissue_6\\mv.inp'
        with open(model_mv_path, 'r') as file:
            lines = file.readlines()

        u_v_data = []
        collect_data = False

        for line in lines:
            if collect_data:
                elements = line.strip().split(' ')
                u_v_data.append([int(elements[0]), float(elements[1]), float(elements[2])])
            elif line.strip() == 'v, none':
                collect_data = True

        return np.array(u_v_data)
    
    def split_u_v_data(self, u_v_data):
        #this sorts the data into different spline strings, 
        #79 strings with 40 points each
        #nr of point, u, v
        split_arrays= np.zeros((79, 40,3))
        index = 0
        for i in range(40):
            for j in range(79):
            
                split_arrays[j,i,:]= u_v_data[index,:]
                index = index + 1

        return split_arrays
    
    def u_v_to_points(self, split_u_v):
        model_mv_path = r'C:\\Users\\ipatzke\\OneDrive - Philips\\Documents\\Masterarbeit\\MasterMission\\Resourcen\\Tissue_6\\mv.inp'
        model_mv_vertices, model_mv_faces = self.load_inp_file(model_mv_path)
        
        point_array = np.zeros((79, 40,3))
        for i in range(40):
            for j in range(79):
                point_array[j,i,:]= model_mv_vertices[int(split_u_v[j,i,0])]

        return point_array
    
    def u_v_to_points_with_color(self, split_u_v):
        model_mv_path = r'C:\\Users\\ipatzke\\OneDrive - Philips\\Documents\\Masterarbeit\\MasterMission\\Resourcen\\Tissue_6\\mv.inp'
        model_mv_vertices, model_mv_faces = self.load_inp_file(model_mv_path)
        
        point_array = np.zeros((79, 40,5))
        for i in range(40):
            for j in range(79):
                point_array[j,i,0:3]= model_mv_vertices[int(split_u_v[j,i,0])]
                point_array[j,i,3:5]= split_u_v[j,i,1:3]

        return point_array
    
    def read_spline_model(self):
        u_v_data = self.load_u_v_model()
        u_v_data_split = self.split_u_v_data(u_v_data)
        spline_model = self.u_v_to_points(u_v_data_split)
        return spline_model, u_v_data_split
    
    def read_colored_spline_model(self):
        u_v_data = self.load_u_v_model()
        u_v_data_split = self.split_u_v_data(u_v_data)
        spline_model = self.u_v_to_points_with_color(u_v_data_split)
        return spline_model, u_v_data_split
        





    