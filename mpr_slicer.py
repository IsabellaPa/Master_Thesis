import numpy as np
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.transform import Rotation as R
from scipy.spatial import procrustes
from vedo import dataurl, precision, Sphere, Volume, Plotter, Mesh, Arrow, Plane, Line, LinearTransform, Ribbon, Cube, Spline, Tube, Cylinder, Points, merge
#import vtk 
from vedo import utils
import vedo.vtkclasses as vtk
from vedo import build_lut
from scipy.linalg import expm, norm
from datetime import datetime
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.spatial import distance
import vedo
from vedo.utils import numpy2vtk, vtk2numpy, OperationNode
from scipy.spatial import KDTree
from vtk.util.numpy_support import numpy_to_vtk
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from scipy.interpolate import splrep, BSpline
import matplotlib.pyplot as plt



class MPR_Slicer():
    def __init__(self, volume_pre_op, volume_intra_op, spline_model_pre, spline_model_intra, annulus_pre_data, annulus_intra_data, excel_writer, dataloader):
        self.dataloader = dataloader

        self.cut_mpr_auto = True
        self.plot = Plotter(axes=0)
        self.plotter_MVOA = Plotter(N=2, sharecam=False)
        self.plotter_coaptation = Plotter()
        self.plotter_unfolding = Plotter()
        self.test_plot = Plotter(N=2)

        self.plotter_coaptation.add_callback("mouse click", self.func)
        
        self.excel_writer = excel_writer
        
        self.cutting_plane_coordi = np.zeros((3,3))
        self.u = None
        self.v = None
        self.w = None

        self.first_time = True

        self.normal_coordi = True


        self.mpr = None
        self.mpr_pre = None
        self.mpr_intra = None
        self.mpr_pre_bent = None
        self.mpr_intra_bent = None
        self.mpr_through_clip = None

        self.annulus_intersection_pre = None
        self.annulus_intersection_intra = None
        self.annulus_intersection_pre_bent = None
        self.annulus_intersection_intra_bent = None


        self.neighbor_mode = False

        self.mpr_width = None
        self.mpr_height = None
        self.mpr_depth = None


        self.min_color_value = 0
        self.max_color_value = 255

        self.cell_id_array = np.zeros((2,3))

        self.volume_cropped = False
        self.mpr_mode_pre = False
        self.pre_or_intra = 'intra'
        self.volume_pre_op= volume_pre_op
        self.volume_intra_op = volume_intra_op

        self.clip_position_intra = None
        self.clip_position_intra_anatomic = None
        self.clip_position_pre = None

        self.clip_lower_point = None
        self.clip_upper_point = None

        self.very_big_plane = False
        
        self.set_volume()


        self.slice_thickness_intra =None
        self.slice_thickness_pre = None

        self.spline_model = spline_model_intra
        self.spline_model_pre = spline_model_pre 
        self.spline_model_intra = spline_model_intra

        
        
        self.top_of_clip = [0,0,10] #append coorinates of new clips, each cloumn coresponds to one clip
        self.bottom_of_clip = [0,0,0]

        self.tab_2D = None
        self.tab_3D = None
        self.tab_2D_bent = None
        self.intervention_helpers = None

        self.theta = 0

        self.mv_model = None
        self.ant_model = None
        self.post_model = None

        self.mv_model_intra = None
        self.ant_model_intra = None
        self.post_model_intra = None
        self.mv_model_pre = None
        self.ant_model_pre = None
        self.post_model_pre = None

        self.apex = None
        self.apex_pre = None
        self.apex_intra = None

        self.annulus_pre_data = annulus_pre_data
        self.annulus_intra_data = annulus_intra_data

        self.vol = None
        self.mpr_plane_mesh = None
        self.p0_vis = None
        self.p1_vis = None
        self.p2_vis = None 
        self.p3_vis = None 
        self.e1_vis = None 
        self.e2_vis = None 
        self.e3_vis = None
        self.e1_world_vis = None 
        self.e2_world_vis = None 
        self.e3_world_vis = None
        self.best_fit_plane = None
        self.bf_e1_vis = None 
        self.bf_e1_vis = None 
        self.bf_e1_vis = None 
        self.gravity_center_annulus_vis= None
        self.mv_model_mesh= None
        self.ant_model_mesh= None 
        self.post_model_mesh = None


        self.all_visible = True
        self.vol_visible =True
        self.mpr_plane_mesh_visible = False
        self.pts_vis_visible = False
        self.e_coordi_vis_visible = False 
        self.best_fit_plane_visible = True
        self.bf_coordi_vis_visible= True
        self.gravity_center_annulus_vis_visible = True
        self.apex_vis_visible = True
        self.mv_model_mesh_visible = False
        self.ant_model_mesh_visible = True
        self.post_model_mesh_visible = True
        self.world_coordi_visible = True
        self.annulus_visible = False
        self.intersection_line_visible = False
        self.apex_coordi_visible = False
        self.bounding_box_visible = True
        self.axis_apex_gravity_center_visible = True
        self.aortic_point_visibile = True
        self.croped_volume_visible = False
        self.clip_visible = True
        

    def set_volume(self):
        if self.mpr_mode_pre == False:
            self.volume = self.volume_intra_op
        elif self.mpr_mode_pre == True:
            self.volume = self.volume_pre_op

    def set_mv_model(self):
        if self.mpr_mode_pre == False:
            self.mv_model = self.mv_model_intra
            self.ant_model = self.ant_model_intra
            self.post_model = self.post_model_intra
            self.slice_thickness = self.slice_thickness_intra
            self.spline_model = self.spline_model_intra
            self.apex = self.apex_intra
        elif self.mpr_mode_pre == True:
            self.mv_model = self.mv_model_pre
            self.ant_model = self.ant_model_pre
            self.post_model = self.post_model_pre
            self.slice_thickness = self.slice_thickness_pre
            self.spline_model = self.spline_model_pre
            self.apex = self.apex_pre
        mv_model_points = np.reshape(np.array([self.mv_model[0]]),(-1,3))
        self.annulus_points = mv_model_points[0:80, :]
        self.create_model_meshes()
        try:
            self.intervention_helpers.disable_enable_buttons()
        except:
            pass
        self.plot.render()

    def change_volume_mode(self, True_False = None):
        if True_False is None:
            self.mpr_mode_pre = not self.mpr_mode_pre
            if self.mpr_mode_pre == True:
                self.pre_or_intra = 'pre'
            elif self.mpr_mode_pre == False:
                self.pre_or_intra = 'intra'
        else:
            self.mpr_mode_pre = True_False

    
    def prepare_visualization(self, visualize):
        #Heart Volume
        if visualize == True:
            self.vol = Volume(self.volume) 
            self.vol.spacing(s = self.slice_thickness)
            vdist_part = (self.max_color_value-self.min_color_value)/7
            self.vol.cmap([(self.min_color_value, 'blue5'),(vdist_part,'blue9'),(vdist_part*2,'green5'), (vdist_part*3,'green8'),(vdist_part*4,'yellow5'), (vdist_part*5,'yellow8'),(vdist_part*6,'red5'), (vdist_part*7,'red1')], alpha=None, vmin=self.min_color_value, vmax=self.max_color_value)

            #Origin of the World Coordinate System
            self.ori_vis = Sphere(pos=(0,0,0), r=3.0, res=24, quads=False, c='r5', alpha=1.0)

            #Bounding Box
            self.bounding_box = self.vol.box()

            self.create_world_coordi()

            self.apex_mesh = Sphere(pos=self.apex, r = 2)

        self.prepare_model()

        #Visualization and Link to slider
        if visualize == True:
            self.plot.show(self.vol,  self.bounding_box, self.ant_model_mesh, self.post_model_mesh, self.best_fit_plane, 
                       self.gravity_center_annulus_vis, self.bf_e1_vis, self.bf_e2_vis, self.bf_e3_vis, 
                       self.bf_e1_vis_apex, self.bf_e2_vis_apex, self.bf_e3_vis_apex, self.aortic_point_proj_vis, 
                       self.apex_mesh, self.e1_world_vis, self.e2_world_vis, self.e3_world_vis, 
                       self.axis_apex_grav_center).interactive().close() #meshgrid_array, self.ori_vis,self.lip_vis


    def prepare_model(self):
        if self.mv_model is not None:
            mv_model_points = np.reshape(np.array([self.mv_model[0]]),(-1,3))
            self.annulus_points = mv_model_points[0:80, :]
            if self.pre_or_intra == 'pre':
                self.apex = self.apex_pre
            elif self.pre_or_intra == 'intra':
                self.apex = self.apex_intra

            #Best fitting Plane of annulus
            self.calculate_best_fit_plane_of_annulus()
        else:
            if self.pre_or_intra == 'pre':
                self.annulus_points = self.annulus_pre_data
            if self.pre_or_intra == 'intra':
                self.annulus_points = self.annulus_intra_data
        
    

        
        if self.spline_model is not None: 
            self.calculate_closure_line_height()

        

        #Corner points and Coordi visualization Preparation
        self.initialize_mpr_helpers()
        self.calculation_LIP()


    def create_model_meshes(self):
        if self.pre_or_intra == 'pre' and self.first_time == True:
            self.mv_model[0] = np.multiply(self.slice_thickness, self.mv_model[0])
            self.ant_model[0] = np.multiply(self.slice_thickness, self.ant_model[0])
            self.post_model[0] = np.multiply(self.slice_thickness, self.post_model[0])
            self.first_time = False

        print('self.slice_thickness: ', self.slice_thickness)

        if self.mv_model is not None:
            self.mv_model_mesh = Mesh(self.mv_model)
        if self.ant_model is not None:
            self.ant_model_mesh = Mesh(self.ant_model)
            self.ant_model_mesh.color('orange')
        if self.post_model is not None:
            self.post_model_mesh = Mesh(self.post_model)
            self.post_model_mesh.wireframe()

    def unpack_all_files(self, all_files):
        self.volume_pre_op = all_files[0] 
        self.volume_intra_op = all_files[1]         
        self.mv_model_pre = all_files[2] 
        self.ant_model_pre = all_files[3] 
        self.post_model_pre = all_files[4]
        self.mv_model_intra = all_files[5] 
        self.ant_model_intra = all_files[6] 
        self.post_model_intra= all_files[7] 
        self.slice_thickness_pre = all_files[8] 
        self.slice_thickness_intra  = all_files[9]
        self.spline_model_pre = all_files[10] 
        self.spline_model_intra = all_files[11]
        self.apex_pre = all_files[12] 
        self.apex_intra = all_files[13]
        self.clip_lower_point = all_files[15]
        self.clip_upper_point = all_files[17]
        
        self.set_mv_model()
        if self.mv_model !=None:
            self.create_model_meshes()

    

    def initialize_mpr_helpers(self):

        self.p0_vis = Sphere(pos=(0,0,0), r=3.0, res=24, quads=False, c='g5', alpha=1.0)
        self.p1_vis = Sphere(pos=(0,0,0), r=3.0, res=24, quads=False, c='g5', alpha=1.0)
        self.p2_vis = Sphere(pos=(0,0,0), r=3.0, res=24, quads=False, c='g5', alpha=1.0)
        self.p3_vis = Sphere(pos=(0,0,0), r=3.0, res=24, quads=False, c='g5', alpha=1.0)


        self.e1_vis = Arrow(start_pt= (0,0,0), end_pt = (20,0,0), s=None, shaft_radius=None, head_radius=None, head_length=None, res=12, c='red', alpha=1.0)
        self.e2_vis = Arrow(start_pt= (0,0,0), end_pt = (0,20,0), s=None, shaft_radius=None, head_radius=None, head_length=None, res=12, c='green', alpha=1.0)
        self.e3_vis = Arrow(start_pt= (0,0,0), end_pt = (0,0,20), s=None, shaft_radius=None, head_radius=None, head_length=None, res=12, c='blue', alpha=1.0)
        
        verts = [(0,0,0), (0,0,0), (0,0,0), (0,0,0)]#[p0, p1, p2, p3]
        cells = [(0,1,3,2)] # cells same as faces
        
        self.mpr_plane_mesh = Mesh([verts, cells])  

    def create_world_coordi(self):
        self.e1_world_vis = Arrow(start_pt= (0,0,0), end_pt = (20,0,0), s=None, shaft_radius=None, head_radius=None, head_length=None, res=12, c='red', alpha=1.0)
        self.e2_world_vis = Arrow(start_pt= (0,0,0), end_pt = (0,20,0), s=None, shaft_radius=None, head_radius=None, head_length=None, res=12, c='green', alpha=1.0) 
        self.e3_world_vis = Arrow(start_pt= (0,0,0), end_pt = (0,0,20), s=None, shaft_radius=None, head_radius=None, head_length=None, res=12, c='blue', alpha=1.0)

    def cut_mpr_at_bounding_box(self, mpr):
        # Find columns with only zeros on the left and right
        trimmed_array = mpr
        if np.all(trimmed_array[0,:] == 0):
            trimmed_array = trimmed_array[1:,:]
            self.top_trim = self.top_trim + 1
        if np.all(trimmed_array[-1,:] == 0):
            trimmed_array = trimmed_array[:-1,:]
            self.bottom_trim = self.bottom_trim + 1

        if np.all(trimmed_array[:, 0] == 0):
            trimmed_array = trimmed_array[:, 1:]
            self.left_trim = self.left_trim + 1
        if np.all(trimmed_array[:, -1] == 0):
            trimmed_array = trimmed_array[:, :-1]
            self.right_trim = self.right_trim + 1
        
        if np.array_equal(trimmed_array, mpr):
            return trimmed_array
        else:
            return self.cut_mpr_at_bounding_box(trimmed_array)

    def is_point_inside_box(self, point, box_dimensions):
        x, y, z = point
        width, height, depth = box_dimensions
        return (0 <= x <= width) and (0 <= y <= height) and (0 <= z <= depth)

        
    
    def slice_plane_custom(self, p0, p1, p2, p3, coordi_ori, cutting_plane_coordi):

        self.p0_scaled = np.multiply(p0, self.slice_thickness)
        self.p1_scaled = np.multiply(p1, self.slice_thickness)
        self.p2_scaled = np.multiply(p2, self.slice_thickness)
        self.p3_scaled = np.multiply(p3, self.slice_thickness)

        coordi_ori_scaled = np.multiply(coordi_ori, self.slice_thickness)

    
        #Moves corner points to new positions
        self.p0_vis.pos(self.p0_scaled)
        self.p1_vis.pos(self.p1_scaled)
        self.p2_vis.pos(self.p2_scaled)
        self.p3_vis.pos(self.p3_scaled)


        #Moves arrow visuals to new position
        #They have to be moved like this, defining a new end points of the arrows is not possible
        #rotate first
        T = np.transpose(cutting_plane_coordi)
        self.e1_vis.apply_transform(T)
        self.e2_vis.apply_transform(T)
        self.e3_vis.apply_transform(T)

        #then translate
        self.e1_vis.pos(coordi_ori_scaled)
        self.e2_vis.pos(coordi_ori_scaled)
        self.e3_vis.pos(coordi_ori_scaled)
        

        verts = [self.p0_scaled, self.p1_scaled, self.p2_scaled, self.p3_scaled]#[p0, p1, p2, p3]
        cells = [(0,1,3,2)] # cells same as faces

        # Build the polygonal Mesh object from the vertices and faces
        self.mpr_plane_mesh.vertices = verts

        rectangle_points = [p0,p1,p2,p3]

        

        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    
    def calculate_meshgrid_and_mpr(self, cutting_plane_coordi, rectangle_points):
        meshgrid_point_array = self.meshgrid_points(cutting_plane_coordi[0, :], cutting_plane_coordi[1, :], rectangle_points[0], 1)

        self.x = np.linspace(0, self.volume.shape[0]-1, self.volume.shape[0])
        self.y = np.linspace(0, self.volume.shape[1]-1, self.volume.shape[1])
        self.z = np.linspace(0, self.volume.shape[2]-1, self.volume.shape[2])
        mpr = interpn((self.x, self.y, self.z), self.volume, meshgrid_point_array, bounds_error=False, fill_value=0)
        mpr = np.transpose(np.reshape(mpr,(self.mpr_width,self.mpr_height)))
        meshgrid_array = Points(np.reshape(meshgrid_point_array,(-1,3)))
        
        self.top_trim = 0
        self.bottom_trim = 0
        self.left_trim = 0
        self.right_trim = 0
        if self.cut_mpr_auto == True:
            mpr_height = mpr.shape[0]
            mpr_width = mpr.shape[1]
            mpr = self.cut_mpr_at_bounding_box(mpr)
            rectangle_points, coordi_ori = self.shrink_mpr_plane_mesh(rectangle_points, mpr_height, mpr_width)
            self.slice_plane_custom(rectangle_points[0], rectangle_points[1], rectangle_points[2], rectangle_points[3], coordi_ori, cutting_plane_coordi)


        tcoords = [[0, 0], [1, 0], [0, 1], [1, 1]]

        self.mpr_plane_mesh.texture(mpr,tcoords = tcoords,interpolate=False,repeat=False, 
                     edge_clamp=False, scale=1, ushift=False, vshift=False)
        
        return mpr
    

    def shrink_mpr_plane_mesh(self, rectangle_points, mpr_height, mpr_width):
        side_1_vector = rectangle_points[2] - rectangle_points[0]
        side_2_vector = rectangle_points[1] - rectangle_points[0]
        middle_of_plane = rectangle_points[0] + 0.5*side_1_vector+0.5* side_2_vector

        half_length_side_1 = 0.5 * np.linalg.norm(rectangle_points[2] - rectangle_points[0])
        half_length_side_2 = 0.5 * np.linalg.norm(rectangle_points[1] - rectangle_points[0])
        half_mpr_height = 0.5* mpr_height
        half_mpr_width = 0.5* mpr_width

        percentage_shrink_top =self.top_trim/mpr_height
        percentage_shrink_bottom =self.bottom_trim/mpr_height
        percentage_shrink_left =self.left_trim/mpr_width
        percentage_shrink_right =self.right_trim/mpr_width

        #shrink top
        rectangle_points[2] = rectangle_points[2]- side_1_vector* percentage_shrink_top
        rectangle_points[3] = rectangle_points[3]- side_1_vector * percentage_shrink_top

        #shrink bottom
        rectangle_points[0] = rectangle_points[0]+ side_1_vector * percentage_shrink_bottom
        rectangle_points[1] = rectangle_points[1]+ side_1_vector * percentage_shrink_bottom

        #shrink left
        rectangle_points[2] = rectangle_points[2]- side_2_vector * percentage_shrink_left
        rectangle_points[0] = rectangle_points[0]- side_2_vector * percentage_shrink_left

        #shrink right
        rectangle_points[1] = rectangle_points[1]- side_2_vector * percentage_shrink_right
        rectangle_points[3] = rectangle_points[3]- side_2_vector * percentage_shrink_right

        side_2_vector_new = rectangle_points[1] - rectangle_points[0]
        coordi_ori = rectangle_points[0] +0.5* side_2_vector_new
        return rectangle_points, coordi_ori

    def reset_helper_points(self):
        
        try:
            T = np.linalg.inv(np.transpose(self.cutting_plane_coordi))
        except:
            T = np.eye(3)

        self.e1_vis.apply_transform(T)
        self.e2_vis.apply_transform(T)
        self.e3_vis.apply_transform(T)

    

    def define_cutting_plane_with_axis(self, position):
        #die Schnittebene ist die x_p-y_p Ebene

        axis_start_p = position[0, :]
        axis_end_p = position[1,:]
        p_in_plane = position[2,:]
        
        cutting_plane_coordi = np.zeros((3,3))
        cutting_plane_coordi[0,:], cutting_plane_coordi[1,:], cutting_plane_coordi[2,:] = self.calculate_eigenvectors_with_point(position)
        rectangle_points = self.calculate_rectangle_points_axis(axis_start_p, axis_end_p, cutting_plane_coordi)

        return cutting_plane_coordi[0,:], cutting_plane_coordi[1,:], cutting_plane_coordi[2,:], rectangle_points

    
    def calculate_eigenvectors_with_point(self, points):
        p0, p1, p2 = points
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        u = self.normalize(np.array([x1-x0, y1-y0, z1-z0]))
        v = self.normalize(np.array([x2-x0, y2-y0, z2-z0]))
        ux, uy, uz = u
        vx, vy, vz = v

        u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]
        normal = np.array(u_cross_v)

        e2 = self.normalize(p1-p0)
        e3 = self.normalize(normal)
        e1 = self.normalize(np.cross(e2,e3))

        e1_vis = Arrow(start_pt=np.multiply(self.slice_thickness,p0), end_pt=np.multiply(self.slice_thickness, p0+e1*50))
        e2_vis = Arrow(start_pt=np.multiply(self.slice_thickness,p0), end_pt=np.multiply(self.slice_thickness, p0+e2*50))
        e3_vis = Arrow(start_pt=np.multiply(self.slice_thickness,p0), end_pt=np.multiply(self.slice_thickness, p0+e3*50))

        p0_vis = Sphere(np.multiply(self.slice_thickness,p0), c ='pink')
        self.plot.add(e1_vis,e2_vis,e3_vis, p0_vis)
        self.plot.render()

        return e1, e2, e3


    
        
    def rotate_plane_slider(self, widget, event):
        self.theta = widget.value - self.theta 
        self.calculate_plane(self.theta)
        self.theta = widget.value

    def calculate_plane(self, angle):
        axis_of_rotation = self.cutting_plane_coordi[1,:]

        self.mpr_plane_mesh.rotate(angle, axis= axis_of_rotation, point=self.axis_start_p)
        self.e1_vis.rotate(angle, axis= axis_of_rotation, point=self.axis_start_p)
        self.e2_vis.rotate(angle, axis= axis_of_rotation, point=self.axis_start_p)
        self.e3_vis.rotate(angle, axis= axis_of_rotation, point=self.axis_start_p)
        self.p0_vis.rotate(angle, axis= axis_of_rotation, point=self.axis_start_p)
        self.p1_vis.rotate(angle, axis= axis_of_rotation, point=self.axis_start_p)
        self.p2_vis.rotate(angle, axis= axis_of_rotation, point=self.axis_start_p)
        self.p3_vis.rotate(angle, axis= axis_of_rotation, point=self.axis_start_p)
        
       
        rotation = R.from_rotvec(np.deg2rad(angle) * axis_of_rotation)
        
        rotated_normal = self.normalize(rotation.apply(self.cutting_plane_coordi[2,:]))
        rotated_e1 = self.normalize(rotation.apply(self.cutting_plane_coordi[0,:]))

        self.cutting_plane_coordi[0,:] = rotated_e1
        self.cutting_plane_coordi[2,:] = rotated_normal

        self.rectangle_points = self.calculate_rectangle_points_axis(self.axis_start_p, self.axis_end_p, self.cutting_plane_coordi)
        mpr = self.calculate_meshgrid_and_mpr(self.cutting_plane_coordi, self.rectangle_points)
        return mpr

        
        
      
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def calculate_clip_orientation(self, clip_number):
        vector_clip = self.top_of_clip[:, clip_number-1] -self.bottom_of_clip[:, clip_number-1]
        angle_clip_cl = self.calculate_angle_between_vectors(vector_clip, self.cl_bf_e1)
        print(f"The angle between the clip and the closure line is: {angle_clip_cl} degrees")
        self.excel_writer.add_value(self.pre_or_intra, 'Clip_orientation', angle_clip_cl)


    def calculate_multiple_clip_distance(self, clip_number_1, clip_number_2):
        vector_clip_1 = self.top_of_clip[:, clip_number_1] -self.bottom_of_clip[:, clip_number_1]
        vector_clip_2 = self.top_of_clip[:, clip_number_2] -self.bottom_of_clip[:, clip_number_2]
        
        center_clip_1 = self.bottom_of_clip[:, clip_number_1] + self.normalize(vector_clip_1) *0.5* np.linalg.norm (vector_clip_1)
        center_clip_2 = self.bottom_of_clip[:, clip_number_2] + self.normalize(vector_clip_2) *0.5* np.linalg.norm (vector_clip_2)

        distance_between_clips = np.linalg.norm(center_clip_1 -center_clip_2)
        return distance_between_clips



    def calculate_clip_intersection(self):
        intersection_lines = self.mv_model_mesh.intersect_with(self.mpr_plane_mesh.triangulate(), tol=1e-06).c('red')
        return intersection_lines
    
    def calculate_clip_intersection_individual(self):
        intersection_line_ant = self.ant_model_mesh.intersect_with(self.mpr_plane_mesh.triangulate(), tol=1e-06).c('red')
        intersection_line_post = self.post_model_mesh.intersect_with(self.mpr_plane_mesh.triangulate(), tol=1e-06).c('red')
        return intersection_line_ant, intersection_line_post 
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
    def calculate_leaflet_scaling(self):
        height_scaled= np.linalg.norm(self.p2_scaled - self.p0_scaled) 
        width_scaled = np.linalg.norm(self.p1_scaled - self.p0_scaled) 

        scaling_x = height_scaled /self.mpr_height
        scaling_y = width_scaled /self.mpr_width
        return scaling_x, scaling_y 


    def normalize(self, vector ):
        normalized_vector = vector / np.sqrt(np.sum(np.float64(vector**2)))
        return normalized_vector
        
    
    def are_vectors_collinear(self, v1, v2):
        # Check if the cross product of the vectors is zero
        cross_product = np.cross(v1, v2)
        return np.allclose(cross_product, 0)
    
    
    def calculate_rectangle_points_axis(self, axis_start_p, axis_end_p, cutting_plane_coordi):
        e1 = cutting_plane_coordi[0, :]
        e2 = cutting_plane_coordi[1, :]

        self.l1= self.extension[1]- self.extension[0]
        self.l2= self.extension[3]- self.extension[2]

        if self.very_big_plane == True:
            p0 = axis_start_p -e1 *self.l1*5
            p1 = axis_start_p +e1 *self.l1*5
            p2 = axis_start_p+e2*self.l2*5 -e1 *self.l1*5
            p3 = axis_start_p+e2*self.l2*5 +e1 *self.l1*5
            rectangle_points = np.array([p0, p1, p2, p3])

        else: 
            p0 = axis_start_p -e1 *self.l1*0.5
            p1 = axis_start_p +e1 *self.l1*0.5
            p2 = axis_start_p+e2*self.l2 -e1 *self.l1*0.5
            p3 = axis_start_p+e2*self.l2 +e1 *self.l1*0.5
            rectangle_points = np.array([p0, p1, p2, p3])

       
        return rectangle_points
    
    def get_automatic_slides(self, number_of_slices, total_angle):
        self.hide_single_mesh()
        try:
            self.hide_set_mesh()
        except:
            pass
        part_angle = total_angle / number_of_slices
        self.all_slices = np.zeros((self.mpr_height, self.mpr_width, number_of_slices))
        

        self.mesh_set_automatic = [Mesh() for i in range(number_of_slices)]
        for i in range(number_of_slices): 
            self.all_slices[:,:,i] = self.calculate_plane(part_angle)
            #Create the according mesh
            p0_scaled = np.multiply(self.rectangle_points[0], self.slice_thickness)
            p1_scaled = np.multiply(self.rectangle_points[1], self.slice_thickness)
            p2_scaled = np.multiply(self.rectangle_points[2], self.slice_thickness)
            p3_scaled = np.multiply(self.rectangle_points[3], self.slice_thickness)
            verts = [p0_scaled, p1_scaled, p2_scaled, p3_scaled]#[p0, p1, p2, p3

            cells = [(0,1,3,2)]
            single_mesh = Mesh([verts, cells])
            tcoords = [[0, 0], [1, 0], [0, 1], [1, 1]]
            single_mesh.texture(self.all_slices[:,:,i],tcoords = tcoords,interpolate=False,repeat=False, edge_clamp=False, scale=1, ushift=False, vshift=False)
            single_mesh.alpha(0.5)
            self.mesh_set_automatic.append(single_mesh)

        self.plot_all_slices()

    def remove_rest(self):
        try:
            self.plot.remove(self.annulus)
            self.plot.remove(self.bf_e1_vis, self.bf_e2_vis, self.bf_e3_vis)
            self.plot.remove(self.bf_e1_vis_apex, self.bf_e2_vis_apex, self.bf_e3_vis_apex)
            self.plot.remove(self.axis_apex_grav_center)
            self.plot.remove(self.e1_vis, self.e2_vis, self.e3_vis)
        except:
            pass
        


    def plot_all_slices(self):
        self.plot.show(self.mesh_set_automatic)

    def hide_single_mesh(self):
        self.plot.remove(self.mpr_plane_mesh)

    def hide_set_mesh(self):
        self.plot.remove(self.mesh_set_automatic)
    
    def remove_all_slices(self):
        self.plot.remove(self.mesh_set_automatic)

    def meshgrid_points(self, e1, e2, p0, spacing):
        
        meshgrid_point_array= np.zeros((self.mpr_width,self.mpr_height,3))
        for i in range(int(self.mpr_width/spacing)):
            for j in range(int(self.mpr_height/spacing)):
                meshgrid_point_array[i,j, :] = p0 +e1*spacing*i + e2*spacing*j
        meshgrid_point_array = np.reshape(meshgrid_point_array, (-1, 3))
        return meshgrid_point_array
    
    def get_clip_mpr_position(self):
        clip_position = np.zeros((3,3))
        clip_lower_point = [150,150,100]
        clip_upper_point = [150,150,110]
        
        clip_position[0,:] = clip_lower_point
        clip_position[1,:] = clip_upper_point
        clip_position[2,:] = self.lip
        self.excel_writer.add_value(self.pre_or_intra, 'Clip_position', clip_position[0:2,:])
        return clip_position

    def set_clip_mpr_position(self, clip_upper_point, clip_lower_point):
        self.clip_lower_point = clip_lower_point
        self.clip_upper_point = clip_upper_point

    
    def get_saved_clip_position(self):
        self.reuse_clip_position()
        return self.clip_position_pre

    
    def calculate_mpr(self, extension, position, visualize):
        #type: defines if plane is defined by points or vectors
        #extension: defines the extension of the plane in 2 dimensions (u0, u1, v0, v1)
        #resolution: defines the desity of points for interpolation
        #position: contains points and a normala vector or angles and one point
        self.position = position
        self.hide_mpr_elements()
        self.reset_helper_points()

        self.extension = extension
        
        self.mpr_width = extension[1]
        self.mpr_height = extension[3]

        self.axis_start_p = position[0, :]
        self.axis_end_p = position[1,:]
        self.p_in_plane = position[2,:]

        self.cutting_plane_coordi[0,:], self.cutting_plane_coordi[1,:], self.cutting_plane_coordi[2,:], self.rectangle_points = self.define_cutting_plane_with_axis(position)
        self.slice_plane_custom(self.rectangle_points[0], self.rectangle_points[1], self.rectangle_points[2], self.rectangle_points[3], self.axis_start_p, self.cutting_plane_coordi)
        mpr = np.transpose(self.calculate_meshgrid_and_mpr(self.cutting_plane_coordi, self.rectangle_points))
        
        spline = Spline(self.annulus_points)
        self.intersection_points_mpr_annulus = np.multiply(np.reciprocal(self.slice_thickness),self.convert_list_to_array(self.get_intersection_with_spline(spline, self.mpr_plane_mesh)))
        
        if self.mpr_mode_pre== True: 
            self.mpr_pre = mpr
            self.annulus_intersection_pre = np.zeros((2,3))
            for i in range(self.intersection_points_mpr_annulus.shape[0]):
                self.annulus_intersection_pre[i,:] = self.transform_world_to_mpr(self.intersection_points_mpr_annulus[i,:])
            
        elif self.mpr_mode_pre== False:
            self.mpr_intra = mpr
            self.annulus_intersection_intra = np.zeros((2,3))
            for i in range(self.intersection_points_mpr_annulus.shape[0]):
                self.annulus_intersection_intra[i,:] = self.transform_world_to_mpr(self.intersection_points_mpr_annulus[i,:])
            

        self.mpr = mpr
        

        if self.tab_3D is not None and visualize == True:
            self.tab_3D.update_mpr(self.mpr_mode_pre, False)

        if self.tab_2D is not None and visualize == True:
            self.tab_2D.update_mpr()

        self.neighbor_mode = False

        
        if self.mv_model is not None:
            self.calculate_leaflet_length_and_intersections()

        if visualize == True:
            self.width_scaled, self.height_scaled= self.calculate_leaflet_scaling()
            self.tab_2D.set_scaling_factor(self.width_scaled, self.height_scaled)
            self.tab_2D.update_scaling()

        self.save_mpr_orientation(position)



        self.clip_vis = Cylinder(pos= self.gravity_center_annulus-self.bf_e3*12-self.bf_e1*3+self.bf_e2*6, r = 2.5, height=10, c = 'red')
        self.plot.add(self.p0_vis, self.p1_vis, self.p2_vis, self.p3_vis, self.e1_vis, self.e2_vis, self.e3_vis, self.mpr_plane_mesh)
        if visualize == True:
            self.show_mpr_plot()


    def save_mpr_orientation(self, position):
        position = np.array(position)
        self.anatomic_position = self.transform_world_to_anatomic(position, self.anatomic_coordi)
        #to test anatomic coordis
        if self.apex.size != 0:
            self.anatomic_coordi_normal = np.array([self.bf_e1_normal, self.bf_e2_normal, self.bf_e3_normal])
            self.anatomic_position_normal = self.transform_world_to_anatomic(position, self.anatomic_coordi_normal)

            self.anatomic_coordi_apex = np.array([self.bf_e1_apex, self.bf_e2_apex, self.bf_e3_apex])
            self.anatomic_position_apex = self.transform_world_to_anatomic(position, self.anatomic_coordi_apex)

    
    def reuse_mpr_orientation(self):
        self.world_position = self.transform_anatomic_to_world(self.anatomic_position, self.anatomic_coordi)

        #to test anatomic coordis
        if self.apex.size != 0:
            self.world_position_normal = self.transform_anatomic_to_world(self.anatomic_position_normal, self.anatomic_coordi_normal)
            self.world_position_apex = self.transform_anatomic_to_world(self.anatomic_position_apex, self.anatomic_coordi_apex)

        return self.world_position
    
    def save_clip_position(self):
        self.clip_position_intra_anatomic = self.transform_world_to_anatomic(self.clip_position_intra, self.anatomic_coordi)

    def reuse_clip_position(self):
        self.clip_position_pre = self.transform_anatomic_to_world(self.clip_position_intra_anatomic, self.anatomic_coordi)


    def calculate_leaflet_length_and_intersections(self):
        self.clip_intersection = self.calculate_clip_intersection()
        self.clip_intersection_ant, self.clip_intersection_post = self.calculate_clip_intersection_individual()
        try:
            self.total_leaflet_length_ant = self.calculate_leaflet_total_length(self.clip_intersection_ant.vertices)
            self.total_leaflet_length_post = self.calculate_leaflet_total_length(self.clip_intersection_post.vertices)
            print('total_leaflet_length_ant: ', self.total_leaflet_length_ant)
            print('total_leaflet_length_post: ', self.total_leaflet_length_post)

            self.excel_writer.add_value(self.pre_or_intra, 'leaflet_length_ant_intersection', np.round(self.total_leaflet_length_ant,2))
            self.excel_writer.add_value(self.pre_or_intra, 'leaflet_length_post_intersection', np.round(self.total_leaflet_length_post,2))
            
        except:
            pass

    
    def calculate_leaflet_total_length(self, way_points):
        
        #This first sorts the points for the clostest neighbors (the points are not always sorted when taken from the intersection line)
        #As it is unknown which point is the first in the spline a loop is created (insert the tube for visualization)
        #Then the longest distance is deleted, as it is known that the distance between the leaftlet tip and hinge has to be way bigger than the distance between any waypoints
        single_curve_lengths = np.zeros(way_points.shape[0])
        way_points = self.nearest_neighbor(way_points)
        
        for i in range(way_points.shape[0]-2):
            single_curve_lengths[i] = np.linalg.norm(way_points[i,:]-way_points[i+1,:])
        single_curve_lengths[-1] = np.linalg.norm(way_points[-1,:]-way_points[0,:])
        single_curve_lengths = np.where(single_curve_lengths == np.max(single_curve_lengths), 0 ,single_curve_lengths)
        
        total_curve_length = np.cumsum(single_curve_lengths)
        length_of_leaflet = total_curve_length[-1]
        return length_of_leaflet
    
    def calculate_leaflet_length_differences(self, length_1, length_2, length_bent_1, length_bent_2):
        diff_1 = length_1- length_bent_1
        diff_2 = length_2- length_bent_2

        if diff_1 > 2:
            print('Caution! The difference for the anterior leaflet is bigger than 2 mm. It might be bent.')
            self.excel_writer.add_value(self.pre_or_intra, 'bent_ant_detected', 1)
        else: 
            self.excel_writer.add_value(self.pre_or_intra, 'bent_ant_detected', 0)
        if diff_2 > 2:
            print('Caution! The difference for the leaflet posterior  is bigger than 2 mm. It might be bent.')
            self.excel_writer.add_value(self.pre_or_intra, 'bent_post_detected', 1)
        else: 
            self.excel_writer.add_value(self.pre_or_intra, 'bent_post_detected', 0)
        print('The difference for leaflet 1: ', diff_1)
        print('The difference for leaflet 2: ', diff_2)

    def show_way_points(self, points, color):
        spline_waypoints = [Sphere() for i in range(points.shape[0])]
        for i in range(points.shape[0]):
            single_sphere = Sphere(pos = points[i,:], c = color, r = 0.3)
            spline_waypoints[i] = single_sphere

        tube = Tube(points, r = 0.1)


    def nearest_neighbor(self, points):
        start_point = 0 
        sorted_indices = [start_point]  # Starting with the first point
        remaining_indices = set(range(1, len(points)))  # Indices of remaining points
        
        current_index = start_point
        
        while remaining_indices:
            nearest_index = min(remaining_indices, key=lambda x: distance.euclidean(points[current_index], points[x]))
            sorted_indices.append(nearest_index)
            remaining_indices.remove(nearest_index)
            current_index = nearest_index
        
        sorted_points = points[sorted_indices]
        sorted_points = np.vstack((sorted_points, sorted_points[0]))
        return sorted_points
   

    def closest_point_to_annulus(self, candidate_points):
        min_distance = 100
        closest_index = None

        for i, point_a in enumerate(self.annulus_points):
            for j, point_b in enumerate(candidate_points):
                distance = np.linalg.norm(point_a - point_b)
                if distance < min_distance:
                    min_distance = distance
                    closest_index = j

        return closest_index
    
    def get_neighbor_mpr(self, angle):
        mpr_neighbor = self.calculate_neighbor_mpr(angle)

        #Only exists for intra but will be shown in the pre image plot
        self.mpr_pre_neighbor = mpr_neighbor
        
        self.mpr = mpr_neighbor
        self.annulus_intersection_pre = np.zeros((2,3))
        for i in range(self.intersection_points_mpr_annulus.shape[0]):
            self.annulus_intersection_pre[i,:] = self.transform_world_to_mpr(self.intersection_points_mpr_annulus[i,:])
        
        

        if self.tab_3D is not None:
            self.tab_3D.update_mpr(True, False)
        
        if self.tab_2D is not None:
            self.tab_2D.update_mpr()

        self.neighbor_mode= True

        self.show_mpr_plot()

    def calculate_neighbor_mpr(self, angle, position = []):
        self.hide_mpr_elements()
        self.reset_helper_points()

        if len(position) ==0:
            position = self.position
        #use old rectangle point to transform it and calculate the new coordinate system
        #rotate around LIP
        transformed_axis_start = self.transform_around_LIP(position[0,:], angle)
        transformed_axis_end = self.transform_around_LIP(position[1,:], angle)
        transformed_point_in_mpr = self.transform_around_LIP(position[2,:], angle)

        position_neighbor_mpr = np.zeros((3,3))
        position_neighbor_mpr[0, :] = transformed_axis_start
        position_neighbor_mpr[1,:] = transformed_axis_end
        position_neighbor_mpr[2,:] = transformed_point_in_mpr

        cutting_plane_coordi = np.zeros((3,3))
        cutting_plane_coordi[0,:], cutting_plane_coordi[1,:], cutting_plane_coordi[2,:], rectangle_points = self.define_cutting_plane_with_axis(position_neighbor_mpr)
        self.slice_plane_custom(rectangle_points[0], rectangle_points[1], rectangle_points[2], rectangle_points[3], transformed_axis_start, cutting_plane_coordi)
        mpr_neighbor = np.transpose(self.calculate_meshgrid_and_mpr(cutting_plane_coordi, rectangle_points))
        return mpr_neighbor
    
    ################################################################################
    
    ############################   Test bent leaflet   #############################
    ################################################################################
    def test_bent_leaflet(self, angle):
        
        if self.mpr_mode_pre == True:
            position = self.get_saved_clip_position()
            print('position: ', position)
        elif self.mpr_mode_pre == False:
            position = self.get_clip_mpr_position()
            print('position: ', position)
        mpr_bent = self.calculate_neighbor_mpr(angle, position)

        if self.mpr_mode_pre == True:
            self.mpr_pre_bent = mpr_bent
            self.annulus_intersection_pre_bent = np.zeros((2,3))
            for i in range(self.intersection_points_mpr_annulus.shape[0]):
                self.annulus_intersection_pre_bent[i,:] = self.transform_world_to_mpr(self.intersection_points_mpr_annulus[i,:])
            
        elif self.mpr_mode_pre == False:
            self.mpr_intra_bent = mpr_bent
            self.annulus_intersection_intra_bent = np.zeros((2,3))
            for i in range(self.intersection_points_mpr_annulus.shape[0]):
                self.annulus_intersection_intra_bent[i,:] = self.transform_world_to_mpr(self.intersection_points_mpr_annulus[i,:])
            
        
        self.mpr = mpr_bent

        if self.tab_3D is not None:
            self.tab_3D.update_mpr(self.mpr_mode_pre, True)
        
        if self.tab_2D_bent is not None:
            self.tab_2D_bent.update_mpr()

        self.show_mpr_plot()


    def transform_around_LIP(self, point, angle):
        lip = np.multiply(self.lip, np.reciprocal(self.slice_thickness))
        rotation_matrix = self.rotate_around_axis(self.cutting_plane_coordi[1,:], angle)
        transformed_point = np.matmul(rotation_matrix,(point -lip)) + lip
        return transformed_point
    
    def transform_around_LIP_no_scaling(self, point, angle):
        rotation_matrix = self.rotate_around_axis(self.cutting_plane_coordi[1,:], angle)
        transformed_point = np.matmul(rotation_matrix,(point -self.lip)) + self.lip
        return transformed_point




    ################################################################################

    #########################   Transformation functions   #########################  
    ################################################################################
    
    
    def transformation(self, e1, e2, e3, orig, pts):
        pneu = np.nan * np.ones((pts.shape[0], 3))
        M = np.array([e1, e2, e3])
        t = orig[:, np.newaxis]
        if pts.size == 3:
            pneu = M @ (pts - t.squeeze()).T
        else:
            for i in range(pts.shape[0]):
                palt = pts[i, :]
                pneu[i, :] = M @ (palt[:, np.newaxis] - t).flatten()
        return pneu

    def transformation_back(self, e1, e2, e3, orig, pts):
        pneu = np.nan * np.ones((pts.shape[0], 3))
        M = np.linalg.inv(np.array([e1, e2, e3]))
        t = orig[:, np.newaxis]
        if pts.size == 3:
             pneu = np.nan * np.ones((3, 1))
             pneu = (M @ pts.T + t.squeeze()).T
        else:
            for i in range(pts.shape[0]):
                palt = pts[i, :]
                pneu[i, :] = (M @ (palt[:, np.newaxis]) + t).flatten()
        return pneu
    
    def transform_mpr_to_world(self, point_to_transform):
        t_point = np.transpose(self.rectangle_points[0])
        transformed_point = self.transformation_back(self.cutting_plane_coordi[0,:], self.cutting_plane_coordi[1,:], self.cutting_plane_coordi[2,:], t_point, point_to_transform)
        return transformed_point
    
    def transform_world_to_mpr(self, point_to_transform):
        t_point = np.transpose(self.rectangle_points[0])
        transformed_point = self.transformation(self.cutting_plane_coordi[0,:], self.cutting_plane_coordi[1,:], self.cutting_plane_coordi[2,:], t_point, point_to_transform)
        return transformed_point
    
    def transform_mpr_spline_to_world(self, point_to_transform, t_point, coordi):
        t_point = np.transpose(t_point)
        base_of_special = coordi.squeeze()

        transformed_point = self.transformation_back(base_of_special[0,:], base_of_special[1,:], base_of_special[2,:], t_point, point_to_transform)
        return transformed_point
    
    def transform_world_to_mpr_spline(self, point_to_transform, t_point, coordi):
        t_point = np.transpose(t_point)
        base_of_special = coordi.squeeze()

        transformed_point = self.transformation(base_of_special[0,:], base_of_special[1,:], base_of_special[2,:], t_point, point_to_transform)
        return transformed_point
    
    def transform_cl_spline_to_world(self, point_to_transform):
        t_point = np.transpose(self.center_of_point_cloud_coordi)
        transformed_point = self.transformation_back(self.e1_point_cloud, self.e2_point_cloud, self.e3_point_cloud, t_point, point_to_transform)
        return transformed_point
    
    def transform_world_to_cl_spline(self, point_to_transform):
        t_point = np.transpose(self.center_of_point_cloud_coordi)
        base_of_special = np.transpose(self.point_cloud_coordi)
        transformed_point = self.transformation(self.e1_point_cloud, self.e2_point_cloud, self.e3_point_cloud, t_point, point_to_transform)
        return transformed_point
    
    def transform_anatomic_to_world(self, point_to_transform, anatomic_coordinate_system):
        t_point = np.transpose(self.gravity_center_annulus)
        transformed_point = self.transformation_back(anatomic_coordinate_system[0,:], anatomic_coordinate_system[1,:], anatomic_coordinate_system[2,:], t_point, point_to_transform)
        return transformed_point
    
    
    def transform_world_to_anatomic(self, point_to_transform, anatomic_coordinate_system):
        t_point = np.transpose(self.gravity_center_annulus)
        transformed_point = self.transformation(anatomic_coordinate_system[0,:], anatomic_coordinate_system[1,:], anatomic_coordinate_system[2,:], t_point, point_to_transform)
        return transformed_point
    
    
    def rotate_around_axis(self, axis, angle_degrees):
        axis= self.normalize(axis)
        axis_x= axis[0]
        axis_y= axis[1]
        axis_z= axis[2]
        angle_radians = np.radians(angle_degrees)
        cos_angle = np.cos(angle_radians)
        sin_angle = np.sin(angle_radians)
        R = np.array([
            [cos_angle + axis_x**2*(1-cos_angle), axis_x*axis_y*(1-cos_angle) - axis_z*sin_angle, axis_x*axis_z*(1-cos_angle) + axis_y*sin_angle],
            [axis_y*axis_x*(1-cos_angle) + axis_z*sin_angle, cos_angle + axis_y**2*(1-cos_angle), axis_y*axis_z*(1-cos_angle) - axis_x*sin_angle],
            [axis_z*axis_x*(1-cos_angle) - axis_y*sin_angle, axis_z*axis_y*(1-cos_angle) + axis_x*sin_angle, cos_angle + axis_z**2*(1-cos_angle)]])
        return R
    
    ################################################################################

  
    
    ###########################   Volume cropping   ################################
    ################################################################################
    
    def calculate_cropped_volume(self, volume_points, plane_normal, plane_point, reverse_crop_direction):
        # Calculate the signed distance of each point to the plane
        
        x_ = np.linspace(0., volume_points.shape[0]-1, volume_points.shape[0])
        y_ = np.linspace(0., volume_points.shape[1]-1, volume_points.shape[1])
        z_ = np.linspace(0., volume_points.shape[2]-1, volume_points.shape[2])
        x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

        x_flat= np.reshape(x, (-1,1))
        y_flat= np.reshape(y, (-1,1))
        z_flat= np.reshape(z, (-1,1))
        coordinates_in_volume = np.hstack((x_flat, y_flat,z_flat))

        distance_from_plane = np.dot(coordinates_in_volume - plane_point, plane_normal)
        if reverse_crop_direction == True:
            distance_from_plane = distance_from_plane*-1


        #[coordinate_x, coordinate_y, coordinate_z, color_value, distance_to_plane]
        volume_with_distance = np.zeros((volume_points.shape[0]* volume_points.shape[1]* volume_points.shape[2], 5))
        volume_with_distance[:,0:3] = coordinates_in_volume
        volume_with_distance[:, 3] = np.reshape(volume_points,(-1))
        volume_with_distance[:, 4] = distance_from_plane

        negative_rows = volume_with_distance[:, 4] < 0
        volume_with_distance[negative_rows, 3] = 0

        cropped_volume = np.reshape(volume_with_distance[:,3],(volume_points.shape[0], volume_points.shape[1], volume_points.shape[2]))

        return cropped_volume
    
    def calculate_plane_normal_from_points(self, p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p1
        
        n = np.cross(v1, v2)
        n_norm = n / np.linalg.norm(n)
        return n_norm
    
    def crop_along_plane(self, reverse_crop_direction):
        self.hide_all_volumes()
        plane_normal = self.calculate_plane_normal_from_points(self.p0_scaled, self.p1_scaled, self.p2_scaled)
        test_sphere = Sphere(self.p0_scaled, r = 5, c= 'pink')
        self.plot.add(test_sphere)
        self.cropped_volume = self.calculate_cropped_volume(self.volume, plane_normal, self.p0_scaled, reverse_crop_direction)
        self.croped_vol = Volume(self.cropped_volume)
        self.croped_vol.spacing(s = self.slice_thickness)
        vdist_part = (self.max_color_value-self.min_color_value)/7
        self.croped_vol.cmap([(self.min_color_value, 'blue5'),(vdist_part,'blue9'),(vdist_part*2,'green5'), (vdist_part*3,'green8'),(vdist_part*4,'yellow5'), (vdist_part*5,'yellow8'),(vdist_part*6,'red5'), (vdist_part*7,'red1')], alpha=None, vmin=self.min_color_value, vmax=self.max_color_value)
            
        self.volume_cropped = True
        self.plot.show(self.croped_vol)

    ################################################################################

    

    ####################   Creation anatomical coordi   ############################
    ################################################################################

    def coordinate_system_plane_known_normal(self, point_in_plane, center_of_coordi, normal_vector):
        bf_e1 = self.normalize(point_in_plane- center_of_coordi)
        bf_e2 = self.normalize(np.cross(bf_e1, normal_vector))
        bf_e3 = normal_vector
        return bf_e1, bf_e2, bf_e3
    
    def coordinate_system_plane_reference_point(self, gravity_center_annulus, apex):
        
        
        bf_e3 = self.normalize(gravity_center_annulus - np.array(apex))
        aortic_point_projected_to_new_plane= self.project_point_onto_plane(self.aortic_point, bf_e3, gravity_center_annulus)
        bf_e1 = self.normalize(aortic_point_projected_to_new_plane - gravity_center_annulus)
        bf_e2 = self.normalize(np.cross(bf_e1, bf_e3))
        return bf_e1, bf_e2, bf_e3
    
    def calculate_angle_between_vectors(self, vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        cosine_similarity = dot_product / (magnitude1 * magnitude2)
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        angle_in_degrees = np.degrees(np.arccos(cosine_similarity))
        
        return angle_in_degrees
    
    def define_anatomic_coordi(self, annulus_points):
        self.gravity_center_annulus = self.polygon_centroid(annulus_points)
        self.aortic_point= annulus_points[40,:]
        

        _, normal_vector = self.calculate_best_fit_plane_normal(annulus_points-self.gravity_center_annulus)

        self.aortic_point_proj = self.project_point_onto_plane(self.aortic_point, normal_vector, self.gravity_center_annulus)

        self.bf_e1_normal, self.bf_e2_normal, self.bf_e3_normal = self.coordinate_system_plane_known_normal(self.aortic_point_proj, self.gravity_center_annulus, normal_vector)

        ori = np.array([0,0,0])

        e1_point = ori + np.array([1,0,0]) *30
        e2_point = ori + np.array([0,1,0]) *30
        e3_point = ori + np.array([0,0,1]) *30
        e1_point_vis = Sphere(e1_point, r = 5, c= 'green')
        e2_point_vis = Sphere(e2_point, r = 5, c= 'blue')
        e3_point_vis = Sphere(e3_point, r = 5, c= 'red')

        transformiert = self.transform_anatomic_to_world(np.array([e1_point, e2_point, e3_point]), np.array([self.bf_e1_normal, self.bf_e2_normal, self.bf_e3_normal]))

        e1t_point_vis = Sphere(transformiert[0], r = 5, c= 'green')
        e2t_point_vis = Sphere(transformiert[1], r = 5, c= 'blue')
        e3t_point_vis = Sphere(transformiert[2], r = 5, c= 'red')

        if self.apex.size != 0:
            self.bf_e1_apex, self.bf_e2_apex, self.bf_e3_apex = self.coordinate_system_plane_reference_point(self.gravity_center_annulus, self.apex)
            
            angle_in_degrees = self.calculate_angle_between_vectors(self.bf_e3_normal, self.bf_e3_apex)
            print(f"The angle between the vectors is: {angle_in_degrees} degrees")
            self.excel_writer.add_value(self.pre_or_intra, 'angle_anatomic_coordi', np.round(angle_in_degrees,2))

            self.anatomic_coordi_normal = np.array([self.bf_e1_normal, self.bf_e2_normal, self.bf_e3_normal])
            self.anatomic_coordi_apex = np.array([self.bf_e1_apex, self.bf_e2_apex, self.bf_e3_apex])

            
        else: 
            self.bf_e1_apex, self.bf_e2_apex, self.bf_e3_apex = None, None, None 

        
        
        return self.bf_e1_normal, self.bf_e2_normal, self.bf_e3_normal, self.bf_e1_apex, self.bf_e2_apex, self.bf_e3_apex

    def test_anatomic_coordi_with_AP(self):
        #erst beide Schnittpunkte bestimmen, dann den auswählen der näher dran ist
        self.normal_coordi == True
        self.calculate_mpr()
        mpr_plane_mesh_normal = self.mpr_plane_mesh
        self.normal_coordi == False
        self.calculate_mpr()
        mpr_plane_mesh_apex = self.mpr_plane_mesh



        #annulus nach der Länge parametrisieren
        #l_partial berechnen
        #Prozent berechnen
        percent_AP = 0
        #Ergebnis speichern
        self.excel_writer.add_value(self.pre_or_intra, 'percent_AP', percent_AP)

    def get_annulus_t_intersection(self):

        spline = Spline(self.annulus_points).lw(3)

        intersection_points_mpr_annulus = self.get_intersection_with_spline(spline, self.mpr_plane_mesh)

        x_spline, y_spline, z_spline, t_param = self.calculate_cubic_spline(self.annulus_points)

        self.t1 = self.find_t_for_point(intersection_points_mpr_annulus[0], self.annulus_points, t_param)
        self.t2 = self.find_t_for_point(intersection_points_mpr_annulus[1], self.annulus_points, t_param)

        t1_sphere = Sphere(intersection_points_mpr_annulus[0], r = 5, c = 'yellow')
        t2_sphere = Sphere(intersection_points_mpr_annulus[1], r = 5, c = 'green')

        return self.t1, self.t2

    def calculate_percent_AP(self):
        self.very_big_plane = True
        apex_plane_coordi = np.zeros((3,3))
        normal_plane_coordi = np.zeros((3,3))

        apex_plane_coordi[0,:], apex_plane_coordi[1,:], apex_plane_coordi[2,:], rectangle_points_apex= self.define_cutting_plane_with_axis(self.world_position_apex)
        normal_plane_coordi[0,:], normal_plane_coordi[1,:], normal_plane_coordi[2,:], rectangle_points_normal= self.define_cutting_plane_with_axis(self.world_position_normal)

        apex_annulus_mesh = self.create_mesh_from_points(rectangle_points_apex, 'red')
        normal_annulus_mesh = self.create_mesh_from_points(rectangle_points_normal, 'cyan')

        spline = Spline(self.annulus_points).lw(3)
        intersection_points_apex_annulus = self.get_intersection_with_spline(spline, apex_annulus_mesh)
        intersection_points_normal_annulus = self.get_intersection_with_spline(spline, normal_annulus_mesh)
        self.very_big_plane = False

        point_pairs= self.closest_pairs([intersection_points_apex_annulus, intersection_points_normal_annulus])

        x_spline, y_spline, z_spline, t_param = self.calculate_cubic_spline(self.annulus_points)

        # Find the parameter values for these points
        t0_apex = self.find_t_for_point(point_pairs[0][0], self.annulus_points, t_param)
        t1_apex = self.find_t_for_point(point_pairs[1][0], self.annulus_points, t_param)
        t0_normal = self.find_t_for_point(point_pairs[0][1], self.annulus_points, t_param)
        t1_normal = self.find_t_for_point(point_pairs[1][1], self.annulus_points, t_param)

        annulus_length=  self.calculate_spline_length_total(self.annulus_points)

        distance_first_pair =  np.abs(t0_apex - t0_normal)
        distance_second_pair =  np.abs(t1_apex - t1_normal)

        distance_t0_apex_original = np.min([np.abs(t0_apex -self.t1), np.abs(t0_apex -self.t2)])
        distance_t1_apex_original = np.min([np.abs(t1_apex -self.t1), np.abs(t1_apex -self.t2)])
        distance_t0_normal_original = np.min([np.abs(t0_normal -self.t1), np.abs(t0_normal -self.t2)])
        distance_t1_normal_original = np.min([np.abs(t1_normal -self.t1), np.abs(t1_normal -self.t2)])

        percentual_distance_1_pair = np.round((distance_first_pair/1)* 100, 2)
        percentual_distance_2_pair = np.round((distance_second_pair/1)*100, 2)

        percentual_distance_t0_apex_original = np.round((distance_t0_apex_original/1)* 100, 2)
        percentual_distance_t1_apex_original = np.round((distance_t1_apex_original/1)*100, 2)
        percentual_distance_t0_normal_original = np.round((distance_t0_normal_original/1)* 100, 2)
        percentual_distance_t1_normal_original = np.round((distance_t1_normal_original/1)*100, 2)
        
        self.excel_writer.add_value(self.pre_or_intra, 'percent_AP_a_n_0', percentual_distance_1_pair)
        self.excel_writer.add_value(self.pre_or_intra, 'percent_AP_a_n_1', percentual_distance_2_pair)
        self.excel_writer.add_value(self.pre_or_intra, 'percent_AP_a_0', percentual_distance_t0_apex_original)
        self.excel_writer.add_value(self.pre_or_intra, 'percent_AP_a_1', percentual_distance_t1_apex_original)
        self.excel_writer.add_value(self.pre_or_intra, 'percent_AP_n_0', percentual_distance_t0_normal_original)
        self.excel_writer.add_value(self.pre_or_intra, 'percent_AP_n_1', percentual_distance_t1_normal_original)
    
    def create_mesh_from_points(self, points, color):
        p0 = points[0]
        p1 = points[1]
        p2 = points[2]
        p3 = points[3]

        p0_scaled = np.multiply(p0, self.slice_thickness)
        p1_scaled = np.multiply(p1, self.slice_thickness)
        p2_scaled = np.multiply(p2, self.slice_thickness)
        p3_scaled = np.multiply(p3, self.slice_thickness)
        
        verts = [p0_scaled, p1_scaled, p2_scaled, p3_scaled]
        cells = [(0,1,3,2)]

        mesh = Mesh([verts, cells]).c(color)

        return mesh

    def closest_pairs(self, points):

        distances = np.zeros((2,2)) # [[p0-p2, p0-p3],[p1-p2, p1-3]]
        p0 = points[0][0]
        p1 = points[0][1]
        p2 = points[1][0]
        p3 = points[1][1]
        distances[0,0] = np.linalg.norm(p0 - p2)
        distances[0,1] = np.linalg.norm(p0 - p3)
        distances[1,0] = np.linalg.norm(p1 - p2)
        distances[1,1] = np.linalg.norm(p1 - p3)

        first_possible_pair_distance_sum = distances[0,0] + distances[1,1] #p0-p2 p1-p3
        second_possible_pair_distance_sum = distances[0,1] + distances[1,0] #p1-p2 p0-p3

        if first_possible_pair_distance_sum > second_possible_pair_distance_sum:
            pairs = [[p1,p2], [p0,p3]]
        else: 
            pairs = [[p0,p2],[p1,p3]]
        return pairs





    def get_intersection_with_spline(self, spline, plane_mesh):


        pts = spline.vertices
        intersection_points = []
        for i in range(spline.npoints-1):
            p = pts[i]
            q = pts[i+1]
            ipts = plane_mesh.intersect_with_line(p, q)
            if len(ipts):
                intersection_points.append(ipts[0])
        
        return intersection_points
    
    def convert_list_to_array(self, lst):
        arr = np.zeros((2, 3))
        for i, elem in enumerate(lst):
            arr[i, :] = elem
        return arr
        


    def calculate_best_fit_plane_normal(self, points):
        cov_matrix = np.cov(points, rowvar = False)
        eigenvalues, eigenvectors = eigh(cov_matrix)
        normal_vector = self.normalize(eigenvectors[:,0])
        return eigenvalues, normal_vector


    def calculate_best_fit_plane_of_annulus(self):
        self.annulus= Tube(self.annulus_points, c = 'red')
        self.plot.add(self.annulus)

        #Schwerpunkt
        bf_e1_normal, bf_e2_normal, bf_e3_normal, bf_e1_apex, bf_e2_apex, bf_e3_apex = self.define_anatomic_coordi(self.annulus_points)
        

        if self.apex.size != 0:
            self.axis_apex_grav_center = Line(self.apex, self.gravity_center_annulus, closed=False, res=2, lw=5, c='red', alpha=1.0)

        self.gravity_center_annulus_vis = Sphere(pos=self.gravity_center_annulus, r=3.0, res=24, quads=False, c='red', alpha=1.0)

        #visualization cooridinate system with normal
        bf_e1_vis_end = self.gravity_center_annulus + 20* bf_e1_normal
        bf_e2_vis_end = self.gravity_center_annulus + 20* bf_e2_normal
        bf_e3_vis_end = self.gravity_center_annulus + 20* bf_e3_normal

        self.bf_e1_vis = Arrow(start_pt= self.gravity_center_annulus, end_pt = bf_e1_vis_end, s=None, shaft_radius=None, head_radius=None, head_length=None, res=12, c='red', alpha=1.0)
        self.bf_e2_vis = Arrow(start_pt= self.gravity_center_annulus, end_pt = bf_e2_vis_end, s=None, shaft_radius=None, head_radius=None, head_length=None, res=12, c='green', alpha=1.0)
        self.bf_e3_vis = Arrow(start_pt= self.gravity_center_annulus, end_pt = bf_e3_vis_end, s=None, shaft_radius=None, head_radius=None, head_length=None, res=12, c='blue', alpha=1.0)

        if self.apex.size != 0:
            #visualization coordinate stytem with apex
            bf_e1_vis_end_apex = self.gravity_center_annulus + 20* bf_e1_apex
            bf_e2_vis_end_apex = self.gravity_center_annulus + 20* bf_e2_apex
            bf_e3_vis_end_apex = self.gravity_center_annulus + 20* bf_e3_apex

            self.bf_e1_vis_apex = Arrow(start_pt= self.gravity_center_annulus, end_pt = bf_e1_vis_end_apex, s=None, shaft_radius=None, head_radius=None, head_length=None, res=12, c='gray', alpha=1.0)
            self.bf_e2_vis_apex = Arrow(start_pt= self.gravity_center_annulus, end_pt = bf_e2_vis_end_apex, s=None, shaft_radius=None, head_radius=None, head_length=None, res=12, c='gray', alpha=1.0)
            self.bf_e3_vis_apex = Arrow(start_pt= self.gravity_center_annulus, end_pt = bf_e3_vis_end_apex, s=None, shaft_radius=None, head_radius=None, head_length=None, res=12, c='gray', alpha=1.0)


        if self.normal_coordi == True or self.apex.size == 0:
            self.bf_e1 = bf_e1_normal
            self.bf_e2 = bf_e2_normal
            self.bf_e3 = bf_e3_normal
        elif self.normal_coordi == False:
            self.bf_e1 = bf_e1_apex
            self.bf_e2 = bf_e2_apex
            self.bf_e3 = bf_e3_apex

        self.anatomic_coordi = np.array([self.bf_e1, self.bf_e2, self.bf_e3])
        
        self.aortic_point_proj_vis = Sphere(pos= self.aortic_point_proj)


        #best fit (bf) plane
        x_max_dist = np.max(self.annulus_points[:,0])-np.min(self.annulus_points[:,0])
        y_max_dist = np.max(self.annulus_points[:,1])-np.min(self.annulus_points[:,1])
        z_max_dist = np.max(self.annulus_points[:,2])-np.min(self.annulus_points[:,2])
        max_dist =  np.max([x_max_dist, y_max_dist, z_max_dist])

        self.bf_plane_exension = max_dist *1.2

        p0_bf_plane = self.gravity_center_annulus -self.bf_e1*0.5*self.bf_plane_exension - self.bf_e2 * 0.5 *self.bf_plane_exension
        p1_bf_plane = self.gravity_center_annulus +self.bf_e1*0.5*self.bf_plane_exension - self.bf_e2 * 0.5 *self.bf_plane_exension
        p2_bf_plane = self.gravity_center_annulus -self.bf_e1*0.5*self.bf_plane_exension + self.bf_e2 * 0.5 *self.bf_plane_exension
        p3_bf_plane = self.gravity_center_annulus +self.bf_e1*0.5*self.bf_plane_exension + self.bf_e2 * 0.5 *self.bf_plane_exension

        verts = [p0_bf_plane, p1_bf_plane, p2_bf_plane, p3_bf_plane]
        cells = [(0,1,3,2)]
        self.best_fit_plane = Mesh([verts, cells])
        self.best_fit_plane.color('lightblue')

    def project_point_onto_plane(self, point, plane_normal, plane_point):
        a, b, c = plane_normal
        x, y, z = point
        x0, y0, z0 = plane_point 

        dot_product = a * (x - x0) + b * (y - y0) + c * (z - z0)
        denominator = a**2 + b**2+ c**2
        x_p = x - dot_product * a / denominator
        y_p = y - dot_product * b / denominator
        z_p = z - dot_product * c / denominator

        projected_point= np.array([x_p, y_p, z_p])
        return projected_point
    
     
    def polygon_centroid(self, vertices):
        px = vertices[:,0]
        py = vertices[:,1]
        pz = vertices[:,2]

        n = vertices.shape[0] -1
        
        #Center of mass of the edges

        sx = sy = sz = slen = 0
        x1 = px[n]
        y1 = py[n]
        z1 = pz[n]
        for i in range(n):
            x2 = px[i]
            y2 = py[i]
            z2 = pz[i]
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            len = np.sqrt(dx**2 + dy**2 + dz**2)
            sx = sx + (x1 + x2)/2*len
            sy = sy + (y1 + y2)/2*len
            sz = sz + (z1 + z2)/2*len
            slen = slen + len
            x1 = x2
            y1 = y2
            z1 = z2
        centroid_x = sx/slen
        centroid_y = sy/slen
        centroid_z = sz/slen

        centroid = np.array([centroid_x, centroid_y, centroid_z])
        return centroid
    
    ################################################################################

    ############################   closure line   ##################################
    ################################################################################

    def calculate_closure_line_height(self):
        self.closure_line_points = self.find_closure_line(self.spline_model)
        self.gravity_center_closure = self.polygon_centroid(self.closure_line_points)
        self.height_closure_line = np.dot(self.gravity_center_closure-self.gravity_center_annulus, self.bf_e3)
        print('Height of the closure line \n (distance between annulus and closure line): ', np.abs(np.round(self.height_closure_line,2)))

        #best fit plane closure line
        _, normal_vector = self.calculate_best_fit_plane_normal(self.closure_line_points-self.gravity_center_annulus)
        self.aortic_point_proj_to_cl = self.project_point_onto_plane(self.aortic_point, normal_vector, self.gravity_center_annulus)
        self.cl_bf_e1, self.cl_bf_e2, self.cl_bf_e3 = self.coordinate_system_plane_known_normal(self.aortic_point_proj_to_cl, self.gravity_center_closure, normal_vector)
        return self.height_closure_line
    
    def calculate_max_curvature_index(self, point_array):
        #https://de.mathworks.com/matlabcentral/fileexchange/69452-curvature-of-a-1d-curve-in-a-2d-or-3d-space
        N = point_array.shape[0]
        dims = point_array.shape[1]
        if dims == 2:
            point_array = np.column_stack((point_array, np.zeros(N))) # Use 3D expressions for 2D as well

        total_arc_length = np.zeros((N,1))
        radius = a = np.full([N, 1], np.nan)
        curvature_vector = a = np.full([N,3], np.nan)
        for i in range(1,N-1):
            p1 = point_array[i, :]
            p0 = point_array[i-1, :]
            p2 = point_array[i+1, :]
            tri_vertices = np.array([p0, p1, p2])
            circumcenter = np.sum(tri_vertices, axis=0) / 3
            radius[i] = np.linalg.norm(p1 - circumcenter)
            if radius[i] == 0:
                radius[i] = radius[i]+0.000000001
            curvature_vector[i, :] = (p1 - circumcenter) / radius[i]
            total_arc_length[i] = total_arc_length[i - 1] + np.linalg.norm(p1 - p0)
        
        total_arc_length[-1] = total_arc_length[-2] + np.linalg.norm(point_array[-1, :] - point_array[-2, :])

        if dims == 2:
            curvature_vector = curvature_vector[:, :2]

        max_curvature_index = np.argmax(radius[1:-1])

        return max_curvature_index
    

    def find_closure_line(self, all_points_of_spline):
        all_spline_tubes = []
        closure_line_points = np.zeros((79,3))  
        for i in range(79):
            points_tube = all_points_of_spline[i,:,:]
            single_tube = Spline(points_tube)
            if  i in range(15,25) or i in range(50,65):
                single_tube.color('orange')
            else: 
                single_tube.color('green')

            half_of_spline = int(np.round(single_tube.nvertices)/2)
            half_spline = single_tube.vertices[half_of_spline:-1,:]
            max_curvature_index = self.calculate_max_curvature_index(half_spline)#single_tube.vertices)
            point_with_max_curvature = single_tube.vertices[half_of_spline+max_curvature_index]
            closure_line_points[i,:] = point_with_max_curvature
            single_tube.pointcolors[half_of_spline+max_curvature_index-4:half_of_spline+max_curvature_index+4] = [255, 0, 0, 255]
            single_tube.pointcolors[half_of_spline-4:half_of_spline+4] = [255, 255, 0, 255]
            all_spline_tubes.append(single_tube)

        mask = np.ones((closure_line_points.shape[0],), dtype=bool)
        mask[15:25] = False  # exclude rows 15-25
        mask[50:65] = False  # exclude rows 50-65

        # apply the mask to the original array
        cut_closure_line_points = closure_line_points[mask]
        
        return cut_closure_line_points
    ################################################################################
    
    #######################    polt, visibility, etc.   ############################
    ################################################################################
    
    def show_entire_volume(self):
        self.hide_all_volumes()
        self.volume_cropped = False
        self.plot.show(self.vol)
        self.vol_visible = True

    def show_volume(self):
        self.hide_all_volumes()
        self.plot.show(self.vol)
        self.vol_visible = True


    def hide_volume(self):
        self.hide_all_volumes()
    
    def color_scale_volume(self, widget, event):
        self.hide_all_volumes()

        self.min_color_value= int(np.round(widget.value))
        vdist_part = (self.max_color_value-self.min_color_value)/7

        if self.volume_cropped == False:
            volume_reduced = np.where(self.volume<self.min_color_value, 0, self.volume)
            self.vol_reduced = Volume(volume_reduced)
            self.vol_reduced.cmap([(self.min_color_value, 'blue5'),(self.min_color_value + vdist_part,'blue9'),(self.min_color_value + vdist_part*2,'green5'), (self.min_color_value + vdist_part*3,'green8'),(self.min_color_value + vdist_part*4,'yellow5'), (self.min_color_value + vdist_part*5,'yellow8'),(self.min_color_value + vdist_part*6,'red5'), (self.min_color_value + vdist_part*7,'red1')], alpha=None, vmin=self.min_color_value, vmax=self.max_color_value)
        
        if self.volume_cropped == True:
            volume_reduced = np.where(self.cropped_volume<self.min_color_value, 0, self.cropped_volume)
            self.vol_reduced = Volume(volume_reduced)
            self.vol_reduced.cmap([(self.min_color_value, 'blue5'),(self.min_color_value + vdist_part,'blue9'),(self.min_color_value + vdist_part*2,'green5'), (self.min_color_value + vdist_part*3,'green8'),(self.min_color_value + vdist_part*4,'yellow5'), (self.min_color_value + vdist_part*5,'yellow8'),(self.min_color_value + vdist_part*6,'red5'), (self.min_color_value + vdist_part*7,'red1')], alpha=None, vmin=self.min_color_value, vmax=self.max_color_value)
            
        self.vol_reduced.spacing(s = self.slice_thickness)
        self.plot.show(self.vol_reduced)

    def hide_all_volumes(self):
        try:
            self.plot.remove(self.vol_reduced)
        except:
            pass
        try:
            self.plot.remove(self.croped_vol)
        except:
            pass
        try:
            self.plot.remove(self.vol)
            self.vol_visible = False
        except:
            pass


    def hide_mpr_elements(self):
        try:
            self.plot.remove(self.p0_vis)
            self.plot.remove(self.p1_vis)
            self.plot.remove(self.p2_vis)
            self.plot.remove(self.p3_vis)
        except:
            pass
        try:
            self.plot.remove(self.e1_vis)
            self.plot.remove(self.e2_vis)
            self.plot.remove(self.e3_vis)
        except:
            pass
        try:
            self.plot.remove(self.mpr_plane_mesh)
        except:
            pass
        try:
            self.plot.remove(self.clip_intersection)
        except:
            pass

    def show_spline_in_3D(self, points, number_of_spline):
        if points.any():        
            #transform the spline points to world coordinates
            transformed_points = self.transform_mpr_to_world(points)
            if number_of_spline == 1:
                color_of_spline = 'red'
                self.spline_3D_1 = Tube(transformed_points, r=1.0, cap=True, res=12, c=color_of_spline, alpha=1.0)
                self.spline_3D_1.scale(s = self.slice_thickness)
                self.plot.show(self.spline_3D_1)
            elif number_of_spline == 2: 
                color_of_spline = 'green'
                self.spline_3D_2 = Tube(transformed_points, r=1.0, cap=True, res=12, c=color_of_spline, alpha=1.0)
                self.spline_3D_2.scale(s = self.slice_thickness)
                self.plot.show(self.spline_3D_2)
            elif number_of_spline == 3: 
                color_of_spline = 'red'
                self.spline_3D_3 = Tube(transformed_points, r=1.0, cap=True, res=12, c=color_of_spline, alpha=1.0)
                self.spline_3D_3.scale(s = self.slice_thickness)
                self.plot.show(self.spline_3D_3)
            elif number_of_spline == 4: 
                color_of_spline = 'green'
                self.spline_3D_4 = Tube(transformed_points, r=1.0, cap=True, res=12, c=color_of_spline, alpha=1.0)
                self.spline_3D_4.scale(s = self.slice_thickness)
                self.plot.show(self.spline_3D_4)
            
        else:
            pass

    def remove_spline_in_3D(self, spline_number):
        try:
            if spline_number == 0:
                self.plot.remove(self.spline_3D_1)
            elif spline_number == 1:
                self.plot.remove(self.spline_3D_2)
            elif spline_number == 2:
                self.plot.remove(self.spline_3D_3)
            elif spline_number == 3:
                self.plot.remove(self.spline_3D_4)
        except:
            pass

    def show_mpr_plot(self):

        self.vol_visible =True
        self.mpr_plane_mesh_visible = True
        self.pts_vis_visible = True
        self.e_coordi_vis_visible = True 
        self.best_fit_plane_visible = True
        self.bf_coordi_vis_visible= True
        self.gravity_center_annulus_vis_visible = True
        self.mv_model_mesh_visible = False
        self.ant_model_mesh_visible = True
        self.post_model_mesh_visible = True
        self.annulus_visible = True
        self.intersection_line_visible = True
        self.apex_coordi_visible = True
        
        self.plot.show(self.vol, self.p0_vis, self.p1_vis, self.p2_vis, self.p3_vis, self.e1_vis, self.e2_vis, self.e3_vis,  self.bounding_box, self.mpr_plane_mesh, self.ant_model_mesh, self.post_model_mesh, self.clip_intersection, 
                       self.best_fit_plane, self.gravity_center_annulus_vis, self.bf_e1_vis, self.bf_e2_vis, self.bf_e3_vis, self.aortic_point_proj_vis).interactive().close() #meshgrid_array, self.ori_vis,

        
    def adjust_transparency(self, item_array, value, volume_TF):
        #ACHTUNG: transparency settings funktionieren nicht mehr wenn self.plot.add_depth_of_field() genutzt wird
        #es gibt keine Fehlermeldung
        if volume_TF: 
            for item in item_array:
                item.alpha([(0, 0), (50, 0),(255,value)])
        else:
            for item in item_array:
                item.alpha(value)
                current_alpha = item.alpha()
        self.plot.render()

    def adjust_visibility(self, item_array, visibility):
        for item in item_array:
            if visibility == True:
                self.plot.remove(item)
            if visibility == False:
                self.plot.add(item)

        self.plot.render()

    ################################################################################

    ################################## area ########################################
    ################################################################################
    def leaflet_area(self):
        leaflet_area = self.mv_model_mesh.area()
        print('leaflet area: ', np.round(leaflet_area, 2), 'mm^2')
        self.excel_writer.add_value(self.pre_or_intra, 'leaflet_area', np.round(leaflet_area,2))

    def calculate_MV_area(self):
        projected_annulus_points = np.zeros_like(self.annulus_points)
        for i in range(self.annulus_points.shape[0]):
            projected_annulus_points[i,:] = self.project_point_onto_plane(self.annulus_points[i,:] , self.bf_e3, self.gravity_center_annulus)
        projected_annulus_points_closed = np.vstack((projected_annulus_points, projected_annulus_points[0,:]))
        line_annulus = Line(projected_annulus_points_closed)
        mesh_annulus_area = line_annulus.join(reset = True).triangulate()
        mesh_area = mesh_annulus_area.area()
        print('mesh_area: ', np.round(mesh_area,2), 'mm^2')
        self.excel_writer.add_value(self.pre_or_intra, 'MV_area', np.round(mesh_area,2))

    def calculate_MVOA(self, normal_of_plane, single_MVOA, distance_of_slice, slice_number, mode, main_axis = False):
        self.plotter_MVOA.at(0).add(self.ant_model_mesh, self.post_model_mesh)
        upper_limit= self.gravity_center_closure
        rim_of_post_leaflet= self.post_model_mesh.vertices[-41:-1,:]
        self.distance_to_lower_limit= self.find_closest_element_to_plane(rim_of_post_leaflet, self.gravity_center_closure, normal_of_plane)
        distance_of_slice = distance_of_slice * self.distance_to_lower_limit
        
        if main_axis == True:
            try:
                normal_of_plane = self.normal_main_axes
            except:
                self.calculate_normal_main_axes_MVOA()
                normal_of_plane = self.normal_main_axes

        try:
            self.mesh_mv_MVOA_copy
        except:
            self.half_cut_mesh()

        #get the different slices of the area
        self.mesh_areas(upper_limit, self.distance_to_lower_limit, normal_of_plane, distance_of_slice, single_MVOA, slice_number, mode)
        self.plotter_MVOA.at(0).show()
        self.plotter_MVOA.at(1).show()
        self.plotter_MVOA.interactive().close()

    def calculate_normal_main_axes_MVOA(self):
        self.cut_model = self.mv_model_mesh.copy()
        self.cut_model.cut_with_plane(origin = self.gravity_center_closure, normal= -1*self.cl_bf_e3)
        lower_point= self.gravity_center_closure - self.distance_to_lower_limit * self.cl_bf_e3
        self.cut_model.cut_with_plane(origin = lower_point, normal= self.cl_bf_e3)


        # Get the points (vertices) of the mesh
        points = self.cut_model.vertices

        # Perform Principal Component Analysis (PCA)
        pca = PCA(n_components=3)
        pca.fit(points)

        # The principal components are the directions of the main axes
        main_axes = pca.components_

        # Visualize the mesh and the main axes
        origin = np.mean(points, axis=0)  # Compute the center of the mesh

      

        self.normal_main_axes = -1 * self.normalize(main_axes[2])        
        # Display the mesh and the principal axes
        #self.plotter_coaptation.show(self.cut_model, main_axis_lines, axes=1, viewup="z", bg='white')
        return self.normal_main_axes
    
    def find_closest_element_to_plane(self, candidate_array, point_in_plane, plane_normal):
        distance_from_plane = np.dot(candidate_array - point_in_plane, plane_normal)
        min_dist = np.min(np.abs(distance_from_plane))
        return min_dist
    
    def find_closest_element_to_point(self, point, candidate_array):
        distance_from_point = np.zeros(candidate_array.shape[0])
        for i in range(candidate_array.shape[0]):
            distance_from_point[i] = np.linalg.norm(point-candidate_array[i,:])
        min_dist_index = np.argmin(np.abs(distance_from_point))
        closest_point = candidate_array[min_dist_index, :]
        return closest_point

    def mesh_areas(self, upper_limit, distance_to_lower_limit, normal_of_plane, distance_of_slice, single_MVOA, slice_number, mode):
        if single_MVOA == True:
            self.single_slice(normal_of_plane, distance_of_slice, upper_limit, mode)
        elif single_MVOA == False:
            self.multiple_slice(upper_limit, distance_to_lower_limit, normal_of_plane, slice_number)

    def remove_old_slices_MVOA(self):
        try:
            self.plotter_MVOA.at(0).remove(self.mslice)
            self.plotter_MVOA.at(1).remove(self.mslice, self.text_MVOA_slinge_slice)
        except:
            pass
        try:
            self.plotter_MVOA.at(0).remove(self.mslices_array)
            self.plotter_MVOA.at(1).remove(self.mslices_array, self.text_MVOA_set_slice)
        except:
            pass

    def half_cut_mesh(self):
        np_array_vertices= np.reshape(np.array([self.mv_model_mesh.vertices]), (-1, 3))
        np_array_cells = np.reshape(np.array([self.mv_model_mesh.cells]), (-1,3))
        part_of_mesh = int(self.mv_model_mesh.nvertices/2) 
        np_array_vertices_cut = np.reshape(np_array_vertices[part_of_mesh:-1, :], (-1,3))
        np_array_cells = np_array_cells - (part_of_mesh+1)
        mask = np.all((np_array_cells >= 0) & (np_array_cells <= part_of_mesh), axis=1)
        filtered_np_array_cells = np_array_cells[mask]
        self.mesh_mv_MVOA_copy = Mesh([np_array_vertices_cut, filtered_np_array_cells])
        return self.mesh_mv_MVOA_copy

    def single_slice(self, normal_of_plane, distance_of_slice, upper_limit, mode):
        self.remove_old_slices_MVOA()
        new_center= self.calculate_parallel_plane(distance_of_slice, normal_of_plane, upper_limit)

        intersection = self.mesh_mv_MVOA_copy.intersect_with_plane(origin=new_center, normal=normal_of_plane)
        slices = intersection.join(reset=True)
        self.mslice = vedo.pointcloud.merge(slices).color('red')


        tri_slice= slices.triangulate()
        area = np.round(tri_slice.area(), 2)
        self.text_MVOA_slinge_slice = "MVOA red contour: "+  str(area) + "mm^2"
        self.excel_writer.add_value(self.pre_or_intra, mode, area)
        self.plotter_MVOA.at(0).add(self.mslice)
        self.plotter_MVOA.at(1).add(self.mslice, self.text_MVOA_slinge_slice)
        
        
    def multiple_slice(self, upper_limit, distance_to_lower_limit, normal_of_plane, slice_number_total):  
        self.remove_old_slices_MVOA()
        color = ['green', 'blue', 'violet', 'red', 'yellow']
        self.mesh_area_steps = [Mesh() for i in range(slice_number_total)]
        self.mslices_array = []
        self.text_MVOA_set_slice = ""
        for slice_number in range(slice_number_total):
            distance_of_slice= (distance_to_lower_limit / (slice_number_total-1)) *slice_number
            new_center= self.calculate_parallel_plane(distance_of_slice, normal_of_plane, upper_limit)

            new_plane= Plane(pos = new_center, normal = normal_of_plane, s= (100,100))

            try:
                intersection = self.mesh_mv_MVOA_copy.intersect_with_plane(origin=new_center, normal=normal_of_plane)
                slices = intersection.join(reset=True)
                mslices = vedo.pointcloud.merge(slices).color(color[slice_number%5])
                tri_slice= slices.triangulate()
                area = np.round(tri_slice.area(), 2)

                text_MVOA_slinge_slice = "MVOA " + color[slice_number%5] + " contour: "+  str(area) + "mm^2\n"
                self.mslices_array.append(mslices)
                self.text_MVOA_set_slice= self.text_MVOA_set_slice + text_MVOA_slinge_slice
            except:
                pass

            
        self.plotter_MVOA.at(0).add(self.mslices_array)
        self.plotter_MVOA.at(1).add(self.mslices_array, self.text_MVOA_set_slice)
            
    def calculate_parallel_plane(self, distance_of_slice, normal_of_plane, upper_limit):
        new_center = upper_limit - normal_of_plane * distance_of_slice
        return new_center

    def calculate_area_single(self, slices):
        for i in range(len(slices)-1):
            slices[i].compute_normals().clean().linewidth(0.1)
            slices[i+1].compute_normals().clean().linewidth(0.1)
            pids =slices[i].boundaries(return_point_ids=True)
            pids0 =  slices[i+1].boundaries(return_point_ids=True)

            pts = Points(slices[i].vertices[pids]).c('red5').ps(10)
            # Create a Label object for all the vertices in the mesh
            labels = slices[i].labels('id', scale=0.3).c('green2')
            pts0 = Points(slices[i+1].vertices[pids0]).c('red5').ps(10)
            # Create a Label object for all the vertices in the mesh
            labels0 = slices[i+1].labels('id', scale=0.3).c('green2')

            ribbon_mesh = Ribbon(slices[i], slices[i+1], closed =False, mode = 0).color('pink')
            self.plot.add(ribbon_mesh)#, pts, labels, pts0, labels0)

    def find_main_axis(self):
        coaptation_area_mesh = self.cut_coaptation_area_mesh()
        points = np.array(coaptation_area_mesh.vertices)
        masses = np.ones(len(points))  # Assume each vertex has unit mass

        # Calculate center of mass
        center_of_mass = np.average(points, axis=0, weights=masses)

        # Translate points to center of mass
        translated_points = points - center_of_mass

        # Calculate inertia tensor
        I = np.zeros((3, 3))
        for i in range(len(points)):
            x, y, z = translated_points[i]
            I[0, 0] += masses[i] * (y**2 + z**2)
            I[1, 1] += masses[i] * (x**2 + z**2)
            I[2, 2] += masses[i] * (x**2 + y**2)
            I[0, 1] -= masses[i] * x * y
            I[0, 2] -= masses[i] * x * z
            I[1, 2] -= masses[i] * y * z

        I[1, 0] = I[0, 1]
        I[2, 0] = I[0, 2]
        I[2, 1] = I[1, 2]

        # Diagonalize the inertia tensor to get principal moments and axes
        eigenvalues, eigenvectors = eigh(I)

        # Sort eigenvalues and eigenvectors for consistency
        sorted_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Visualize the principal axes
        origin = vedo.Point(center_of_mass, r=12, c='red')
        axes = vedo.Axes(coaptation_area_mesh, xtitle='X', ytitle='Y', ztitle='Z')

        # Create lines for the principal axes
        principal_axes = []
        for i in range(3):
            direction = eigenvectors[:, i]
            start_point = center_of_mass - direction * 200
            end_point = center_of_mass + direction * 200
            line = vedo.Line(start_point, end_point, c='red')
            principal_axes.append(line)

        self.main_axis 
        # Display the mesh and principal axes
        plotter = vedo.Plotter()
        plotter.show(coaptation_area_mesh, origin, axes, *principal_axes, "Principal Axes of the Mesh")


    
    
    ################################################################################

    ############################   Coaptation area   ###############################
    ################################################################################
    def costume_distance_cells(self, mesh_1, mesh_2):
        data_2= np.array(mesh_2.cell_centers)
        sample_1= np.array(mesh_1.cell_centers)
        kdtree=KDTree(data_2)
        dist_1,points_2=kdtree.query(sample_1,1)
        dist_1 = np.reshape(dist_1,(-1,1))

        data_1= np.array(mesh_1.cell_centers)
        sample_2= np.array(mesh_2.cell_centers)
        kdtree=KDTree(data_1)
        dist_2,points_1=kdtree.query(sample_2,1)
        dist_2 = np.reshape(dist_2,(-1,1))
        ##################################################
        mesh_1.compute_normals(points =False, feature_angle=90)
        mesh_2.compute_normals(points =False, feature_angle=90)
        normals_1 = mesh_1.cell_normals
        normals_2 = mesh_2.cell_normals

        index_array_1= [points_1.flatten()]
        index_array_2= [points_2.flatten()]

        #Mesh 1
        scalar_1=np.zeros(normals_1.shape[0])
        for i in range(normals_1.shape[0]-1):
            scalar_1[i] = np.dot(normals_1[i,:], normals_2[index_array_2[0][i],:])
        
        #Mesh 2
        scalar_2=np.zeros(normals_2.shape[0])
        for i in range(normals_2.shape[0]-1):
            scalar_2[i] = np.dot(normals_2[i,:], normals_1[index_array_1[0][i],:])

        grenzwert = -0.6
        weight_1= np.where(scalar_1< grenzwert, 0, (scalar_1-grenzwert)*2)
        weight_2= np.where(scalar_2< grenzwert, 0, (scalar_2-grenzwert)*2)
        weight_1 = np.reshape(weight_1,(-1,1))
        weight_2 = np.reshape(weight_2,(-1,1))

        dist_1 = weight_1 +dist_1
        dist_2 = weight_2 +dist_2

        collected = np.zeros((points_2.shape[0], 7))
        collected[:,0] =dist_1.flatten() 
        collected[:,1:4]= data_2[points_2.flatten()]
        collected[:,4:7]=sample_1
        collected_filtered = collected[collected[:, 0] <= 5]

        dist_1 = np.where(dist_1 > 3, 8, dist_1)
        dist_2 = np.where(dist_2 > 3, 8, dist_2)

        dist_1[0:int(np.round((dist_1.shape[0]*0.3))),:] = np.max(dist_1) #setze 0 um anzuzeigen
        dist_2[0:int(np.round((dist_2.shape[0]*0.3))),:] = np.max(dist_2)

        color = ['violet', 'blue', 'lightblue', 'pink', 'red', 'orange', 'yellow','white']
        for i in range(len(collected_filtered)):
            start_point = collected_filtered[i, 4:7]
            end_point = collected_filtered[i, 1:4]

            dist_abs = np.min([int(np.round(collected_filtered[i,0])),7])
            arrow = vedo.Arrow(start_point, end_point, c='blue').color(color[dist_abs])
            #plotter.add(arrow)

        
        for i in range(len(collected_filtered)):
            dist_abs = np.min([int(np.round(collected_filtered[i,0])),7])
            # Extract coordinates for the two points
            dist_point= Sphere(pos= collected_filtered[i,4:7], r= 6).color(color[dist_abs])
            #plotter.add(dist_point)

        for i in range(normals_1.shape[0]):
            start_point= data_1[i,:]
            end_point = start_point + 1*normals_1[i,:]
            single_arrow = vedo.Arrow(start_point, end_point, c='blue').color(color[dist_abs])
            #plotter.add(single_arrow)

        for i in range(normals_2.shape[0]):
            start_point= data_2[i,:]
            end_point = start_point + 1*normals_2[i,:]
            single_arrow = vedo.Arrow(start_point, end_point, c='blue').color(color[dist_abs])
            #plotter.add(single_arrow)
        #################################

        scals_1 = utils.numpy2vtk(dist_1)
        scals_2 = utils.numpy2vtk(dist_2)   

        
        scals_1.SetName("Distance")
        mesh_1.dataset.GetCellData().AddArray(scals_1)
        mesh_1.dataset.GetCellData().SetActiveScalars(scals_1.GetName())
        rng = scals_1.GetRange()
        mesh_1.mapper.SetScalarRange(rng[0], rng[1])
        mesh_1.mapper.ScalarVisibilityOn()

        scals_2.SetName("Distance")
        mesh_2.dataset.GetCellData().AddArray(scals_2)
        mesh_2.dataset.GetCellData().SetActiveScalars(scals_2.GetName())
        rng = scals_2.GetRange()   
        mesh_2.mapper.SetScalarRange(rng[0], rng[1])
        mesh_2.mapper.ScalarVisibilityOn()
        
        mesh_1.pipeline = utils.OperationNode(
                "distance_to",
                parents=[mesh_1, mesh_2],
                shape="cylinder",
                comment=f"#pts {mesh_1.dataset.GetPointData()}",
            )
        mesh_2.pipeline = utils.OperationNode(
                "distance_to",
                parents=[mesh_2, mesh_1],
                shape="cylinder",
                comment=f"#pts {mesh_2.dataset.GetPointData()}",
            )
        
    def color_mesh_with_distance_values(self):
        self.model_ant_mesh_coaptation_copy = self.ant_model_mesh.copy()
        self.model_post_mesh_coaptation_copy = self.post_model_mesh.copy().wireframe(False)

        self.costume_distance_cells(self.model_ant_mesh_coaptation_copy, self.model_post_mesh_coaptation_copy)

        self.model_ant_mesh_coaptation_copy.cmap(('RdYlGn'))
        self.model_post_mesh_coaptation_copy.cmap(('RdYlGn'))

        color_array_1 = self.smooth_mesh_color(self.model_ant_mesh_coaptation_copy)
        color_array_1[:,3] = 255
        self.model_ant_mesh_coaptation_copy.pointcolors= color_array_1
        self.model_ant_mesh_coaptation_copy.alpha(1)
        color_array_2 = self.smooth_mesh_color(self.model_post_mesh_coaptation_copy)
        color_array_2[:,3] = 255
        self.model_post_mesh_coaptation_copy.pointcolors= color_array_2
        self.model_post_mesh_coaptation_copy.alpha(1)

        self.model_mv_mesh_coaptation = merge(self.model_ant_mesh_coaptation_copy, self.model_post_mesh_coaptation_copy)

        self.plotter_coaptation.show(self.model_mv_mesh_coaptation)
    def smooth_mesh_color(self, mesh):
        nr_vertex_ids = mesh.vertices.shape[0]
        mean_color_array = np.zeros((mesh.vertices.shape[0], 4))
        # Find neighboring faces to the vertex
        for id in range(nr_vertex_ids-1):
            neighbor_faces = []
            neighbor_faces = mesh.connected_cells(id, return_ids=True)
            face_colors = np.array([mesh.cellcolors[i] for i in neighbor_faces])
            for color in range(len(face_colors)):
                mean_color_array[id, :]= np.sum(face_colors, axis=0)/len(face_colors)
        return mean_color_array
    
    def calculate_spline_param(self, spline_points):
        single_curve_lengths = np.zeros(spline_points.shape[0])
        for i in range(spline_points.shape[0]-1):
            single_curve_lengths[i+1] = np.linalg.norm(spline_points[i,:]-spline_points[i+1,:])
        total_curve_length = np.cumsum(single_curve_lengths)
        t = total_curve_length/(total_curve_length[-1])

        return t
    
    def calculate_cubic_spline(self, way_points):
        t = self.calculate_spline_param(way_points)
       
        fx_cs = CubicSpline(t, way_points[:,0])
        fy_cs = CubicSpline(t, way_points[:,1])
        fz_cs = CubicSpline(t, way_points[:,2])
        var = np.linspace(0, 1, num=100)
        x_cs = fx_cs(var)
        y_cs = fy_cs(var)
        z_cs = fz_cs(var)
        return x_cs, y_cs, z_cs, t
    
    def find_t_for_point(self, point, spline_points, t_array):
        distances = []
        for i in range(spline_points.shape[0]):
            spline_point = spline_points[i,:]
            distance = np.linalg.norm(spline_point - point)
            distances.append(distance)
        return t_array[np.argmin(distances)]
    
    def calculate_spline_length(self, spline):
        dist = np.zeros(40)
        for i in range(40-1):
            dist[i+1] = np.linalg.norm(spline[i,:]-spline[i+1,:])
        total_length= np.cumsum(dist)
        return total_length
    
    def fit_spline_between_points_to_mesh(self, intersection_points, point1, point2):
        x_spline, y_spline, z_spline, t_param = self.calculate_cubic_spline(intersection_points.vertices)
        p = np.column_stack((x_spline, y_spline, z_spline))
        

        # Find the parameter values for these points
        t1 = self.find_t_for_point(point1, intersection_points.vertices, t_param)
        t2 = self.find_t_for_point(point2, intersection_points.vertices, t_param)

        start_point = point1
        end_point = point2

        if t1 > t2:
            t1, t2 = t2, t1
            start_point, end_point = end_point, start_point

        spline_points_with_t = np.zeros((intersection_points.vertices.shape[0], 4))
        spline_points_with_t[:, 0:3] =intersection_points.vertices
        spline_points_with_t[:, 3] =t_param

        mask = (spline_points_with_t[:, -1] > t1) & (spline_points_with_t[:, -1] < t2)
        filtered_spline_points = spline_points_with_t[mask]

        spline_points = np.vstack((start_point, filtered_spline_points[:,0:3], end_point))

        spline_vis = Spline(spline_points).color('white')
        return spline_points, spline_vis

    
    def measure_coaptation_height(self):
        if not np.any(self.cell_id_array):
            print('No points have been selected to measure coaptation. Please select two points on the mesh.')
            pass
        else:
            point3 = self.cell_id_array[0,:] + 10 * self.bf_e1
            points = [self.cell_id_array[0,:], self.cell_id_array[1,:], point3]
            e1, e2, e3 = self.calculate_eigenvectors_with_point(points)
            outline = self.model_mv_mesh_coaptation.intersect_with_plane(origin = self.cell_id_array[0,:], normal = e3).color('black')

            outline = outline.join(reset = True)

            spline_points, self.coapt_height_spline_vis = self.fit_spline_between_points_to_mesh(outline, self.cell_id_array[0,:], self.cell_id_array[1,:])
            self.coaptation_height = self.calculate_spline_length_total(spline_points)
        
            print('Coaptation height: ', np.round(self.coaptation_height,2), 'mm')
            self.excel_writer.add_value(self.pre_or_intra, 'coaptation_height', np.round(self.coaptation_height,2))
            self.plotter_coaptation.add(self.coapt_height_spline_vis)
  
    def measure_coaptation_width(self):
        if not np.any(self.cell_id_array):
            print('No points have been selected to measure coaptation. Please select two points on the mesh.')
            pass
        else:
            point3 = self.cell_id_array[0,:] + 10 * self.bf_e1
            test_sphere2 = Sphere(pos = point3, c= 'cyan')
            points = [self.cell_id_array[0,:], self.cell_id_array[1,:], point3]
            e1, e2, e3 = self.calculate_eigenvectors_with_point(points)
            outline = self.model_mv_mesh_coaptation.intersect_with_plane(origin = self.cell_id_array[0,:], normal = e3).color('black')

            outline = outline.join(reset = True)

            spline_points, self.coapt_width_spline_vis = self.fit_spline_between_points_to_mesh(outline, self.cell_id_array[0,:], self.cell_id_array[1,:])
            self.coaptation_width = self.calculate_spline_length_total(spline_points)
        
            print('Coaptation width: ', np.round(self.coaptation_width,2), 'mm')
            self.excel_writer.add_value(self.pre_or_intra, 'coaptation_width', np.round(self.coaptation_width,2))
            self.plotter_coaptation.add(self.coapt_width_spline_vis)

    def measure_coaptation_width_curvy(self):
        if not np.any(self.cell_id_array):
            print('No points have been selected to measure coaptation. Please select two points on the mesh.')
            pass
        else:
            point3 = self.cell_id_array[0,:] + 10 * self.bf_e1
            test_sphere2 = Sphere(pos = point3, c= 'cyan')

            #Make a spline in the shape of the closure line
            cov_matrix = np.cov(self.closure_line_points, rowvar = False)
            eigenvalues, eigenvectors = eigh(cov_matrix)
            vector_flat_side_cl = self.normalize(eigenvectors[:, 1])

            dot_products = np.dot(self.closure_line_points, vector_flat_side_cl)
            # subtract dot product from each point to move it along the normal vector
            projected_cl_points = self.closure_line_points -  dot_products[:, np.newaxis] *  vector_flat_side_cl[np.newaxis, :]

            projected_cl_points_vis = Points(projected_cl_points)
            cl_points = Points(self.closure_line_points).c('red5')
             
            

            
            closure_line_shape = self.lay_spline_through_point_cloud(projected_cl_points, vector_flat_side_cl)
            projected_cl_point = self.project_point_onto_plane(self.gravity_center_closure, vector_flat_side_cl, projected_cl_points[0,:])

            spline_1 = Spline(closure_line_shape)
            distance_to_closure_line = np.linalg.norm((projected_cl_point -  self.gravity_center_closure))
            spline_points_2 = closure_line_shape + vector_flat_side_cl * 2* distance_to_closure_line
            spline_2 = Spline(spline_points_2)
            ribbon = Ribbon(spline_1, spline_2, alpha = 0.2)

            outline = self.model_mv_mesh_coaptation.intersect_with(ribbon).c('cyan')
            outline = outline.join(reset = True)

            outline_tube = Spline(outline)

            #Berechnung von der Closureline length
            self.calculate_closure_line_length(outline)

            selected_point1 = self.cell_id_array[0,:]
            selected_point2 = self.cell_id_array[1,:]

            selected_point_intersection1 = self.find_closest_point_on_spline(outline_tube, selected_point1)
            selected_point_intersection2 = self.find_closest_point_on_spline(outline_tube, selected_point2)

            spline_points, self.coapt_width_spline_cl_vis = self.fit_spline_between_points_to_mesh(outline, selected_point_intersection1, selected_point_intersection2)
            self.coaptation_width_cl = self.calculate_spline_length_total(spline_points)
        
            print('Coaptation width closure line: ', np.round(self.coaptation_width_cl,2), 'mm')
            self.excel_writer.add_value(self.pre_or_intra, 'coaptation_width_closure_line', np.round(self.coaptation_width_cl,2))
            self.plotter_coaptation.add(self.coapt_width_spline_cl_vis)


    def lay_spline_through_point_cloud(self, points, plane_normal):
        projected_cl_point = self.project_point_onto_plane(self.gravity_center_closure, plane_normal, points[0,:])
        projected_annulus_point = self.project_point_onto_plane(self.gravity_center_annulus, plane_normal, points[0,:])


        self.e1_point_cloud = self.normalize(projected_annulus_point-projected_cl_point)
        self.e3_point_cloud = self.normalize(plane_normal)
        self.e2_point_cloud = - 1* self.normalize(np.cross(self.e1_point_cloud, self.e3_point_cloud))
        self.point_cloud_coordi = [self.e1_point_cloud, self.e2_point_cloud, self.e3_point_cloud]

        self.center_of_point_cloud_coordi = projected_cl_point

        e1_vis_end = self.center_of_point_cloud_coordi + 20* self.e1_point_cloud
        e2_vis_end = self.center_of_point_cloud_coordi + 20* self.e2_point_cloud
        e3_vis_end = self.center_of_point_cloud_coordi + 20* self.e3_point_cloud

        e1_point_cloud_vis = Arrow(start_pt=self.center_of_point_cloud_coordi, end_pt= e1_vis_end, c= 'red')
        e2_point_cloud_vis = Arrow(start_pt=self.center_of_point_cloud_coordi, end_pt= e2_vis_end, c= 'green')
        e3_point_cloud_vis = Arrow(start_pt=self.center_of_point_cloud_coordi, end_pt= e3_vis_end, c= 'blue')

        proj_point_annulus_vis = Sphere(projected_annulus_point, c= 'red')
        proj_point_cl_vis = Sphere(projected_cl_point, c = 'green')


        vis_points = Points(points, r = 30, c = 'green', alpha = 0.5)
        points_on_plane = self.transform_world_to_cl_spline(points)
        transformed_point = Points(points_on_plane, r = 20, c ='blue', alpha = 0.5)

        points_on_plane = points_on_plane[points_on_plane[:, 1].argsort()]


        unique_y, indices = np.unique(points_on_plane[:, 1], return_index=True)
        mean_x = np.array([np.mean(points_on_plane[points_on_plane[:, 1] == y, 0]) for y in unique_y])

        # create the new array with unique x values and mean y values
        cleaned_points_on_plane = np.column_stack((mean_x, unique_y))


        x_spline = cleaned_points_on_plane[:,0]
        y_spline = cleaned_points_on_plane[:,1]

        left_limit = np.min(y_spline)
        left_points = np.zeros((20,3))
        for i in range(20):
            new_point =projected_cl_point + (i+1)* self.e2_point_cloud
            new_point = self.project_point_onto_plane(new_point, plane_normal, points[0,:])
            new_point = self.transform_world_to_cl_spline(new_point)
            new_point = new_point-[0,left_limit,0]
            left_points[i,:] =new_point

        left_points_world = self.transform_cl_spline_to_world(left_points)
        
        left_points_world_vis = Points(left_points_world, c= 'red', r = 5)
        left_points_vis = Points(left_points, c= 'red', r = 5)

        right_limit = np.max(y_spline)
        right_points = np.zeros((20,3))
        for i in range(20):
            new_point = projected_cl_point - (i+1) * self.e2_point_cloud
            new_point = self.project_point_onto_plane(new_point, plane_normal, points[0,:])
            new_point = self.transform_world_to_cl_spline(new_point)
            new_point = new_point -[0,right_limit,0]
            right_points[i,:] =new_point
        right_points = np.flip(right_points, axis=0)

        
        right_points_world = self.transform_cl_spline_to_world(right_points)
        right_points_vis = Points(right_points, c='blue', r = 5)
        right_points_world_vis = Points(right_points_world, c='blue', r = 5)


        trans_all_points = np.concatenate((left_points[:, 0:2], cleaned_points_on_plane, right_points[:, 0:2]))

        x_values_for_spline = np.concatenate((right_points[:,1], cleaned_points_on_plane[:,1], left_points[:,1]))
        y_values_for_spline = np.concatenate((right_points[:,0], cleaned_points_on_plane[:,0], left_points[:,0]))

        tck_all_points_20 = splrep(x_values_for_spline, y_values_for_spline, s=20)
        y_all_points_20 = BSpline(*tck_all_points_20)(x_values_for_spline)
        
        
        spline_points_all = np.column_stack((y_all_points_20, x_values_for_spline, np.zeros_like(y_all_points_20)))
        spline_points_trans_all = self.transform_cl_spline_to_world(spline_points_all)

        test_spline_tube = Tube(spline_points_trans_all, r = 0.2, c = 'pink', alpha = 0.7)

        proj_point_annulus_vis = Sphere(projected_annulus_point, c= 'red')
        proj_point_cl_vis = Sphere(projected_cl_point, c = 'green')


        end_pt_e1 = self.gravity_center_closure + self.cl_bf_e1 * 20
        end_pt_e2 = self.gravity_center_closure + self.cl_bf_e2 * 20
        end_pt_e3 = self.gravity_center_closure + self.cl_bf_e3 * 20
        self.cl_bf_e1_vis = Arrow(start_pt= self.gravity_center_closure, end_pt = end_pt_e1, res=12, c='red', alpha=1.0)
        self.cl_bf_e2_vis = Arrow(start_pt= self.gravity_center_closure, end_pt = end_pt_e2, res=12, c='green', alpha=1.0)
        self.cl_bf_e3_vis = Arrow(start_pt= self.gravity_center_closure, end_pt = end_pt_e3, res=12, c='blue', alpha=1.0)

        test_vis = Points(self.center_of_point_cloud_coordi).c('green')
        return spline_points_trans_all

    def find_closest_point_on_spline(self, outline, selected_point):
        plane = Plane(pos= selected_point, normal = self.cl_bf_e2, s = (50,50))
        test_point_select = Sphere(selected_point, c= 'pink', r = 0.3)
        points = self.get_intersection_with_spline(outline, plane)

        #There might be no intersection if a point at the outer parts was chosen
        if len(points) == 0:
            points = self.find_closest_element_to_point(selected_point, outline.vertices)
            test_sphere = Sphere(points, c = 'cyan', r = 0.2)

        elif len(points) > 1:
            points = np.reshape(np.array(points),(-1,3))
            distances = np.zeros(points.shape[0])
            path = []

            colors= ['pink', 'green', 'yellow', 'black']
            for i in range(points.shape[0]):
                test_sphere = Sphere(points[i,:], c = 'cyan', r = 0.3)
                single_path = self.model_mv_mesh_coaptation.geodesic(selected_point, points[i,:]).color(colors[i])
                path.append(single_path)
                if single_path.nvertices !=0:
                    distances[i] = self.calculate_spline_length_total(single_path.vertices)
                else:
                    distances[i] = np.inf
            
            min_dist_index = np.argmin(distances)
            points = points[min_dist_index,:]


        return points


    def calculate_spline_length_total(self, spline):
        dist = np.zeros(spline.shape[0])
        for i in range(spline.shape[0]-1):
            dist[i+1] = np.linalg.norm(spline[i,:]-spline[i+1,:])
        total_length= np.cumsum(dist)
        return total_length[-1]
    
    def visualize_parametrization(self):
        spline_model, u_v_data_split = self.dataloader.read_colored_spline_model()

        all_tubes_u= []
        all_tubes_v= []

        for i in range(79):

            line_u = Line(spline_model[i,:,0:3])
            line_u.pointdata["my_scalars"] = spline_model[i,:,3]
            single_spline_u = Tube(line_u, r=0.3)
            single_spline_u.cmap("rainbow", "my_scalars", vmin = -np.pi, vmax= np.pi)

            line_v = Line(spline_model[i,:,0:3])
            line_v.pointdata["my_scalars"] = spline_model[i,:,4]
            single_spline_v = Tube(line_v, r=0.3)
            single_spline_v.cmap("rainbow", "my_scalars", vmin = 0, vmax= 1)

            all_tubes_u.append(single_spline_u)
            all_tubes_v.append(single_spline_v)



    def create_unfolding(self):
        spline_model, u_v_data_split = self.dataloader.read_spline_model()
        np_array_ant = np.array(self.model_ant_mesh_coaptation_copy.vertices).reshape(-1,3)
        np_array_post = np.array(self.model_post_mesh_coaptation_copy.vertices).reshape(-1,3)
        all_tube_lengths = np.zeros((79, 40))
        tube_color = np.zeros((79, 40, 4))

        for i in range(79):
            single_spline = Spline(spline_model[i,:,:])
            all_tube_lengths[i,:] = self.calculate_spline_length(spline_model[i,:,:])


            for j in range(40):
                point_in_spline = spline_model[i,j,:]
                #set the type equal as the comparison (np.where) will not work otherwise
                point_in_spline = np.asarray(point_in_spline, dtype=np_array_post.dtype)
                #get the index of the point in the mesh
                #same as np.where
                index= (np_array_ant==point_in_spline).all(axis=1).nonzero()[0] 
                try: 
                    index[0]
                    if index.size > 1:
                        index = index[0]
                    #The list is not empty -> point is in the ant mesh
                    tube_color[i,j,:] = self.model_ant_mesh_coaptation_copy.pointcolors[index]
                except IndexError: 
                    #Empty List -> point has to be in the post mesh
                    index= (np_array_post==point_in_spline).all(axis=1).nonzero()[0]
                    if index.size > 1:
                        index = index[0]
                    tube_color[i,j,:] = np.reshape(self.model_post_mesh_coaptation_copy.pointcolors[index,:], (1, 4))
                
            self.plotter_unfolding.add(single_spline)

        plot_data= np.zeros((79,40,2))
        plot_data[:,:,0]= u_v_data_split[:,:,1]
        plot_data[:,:,1]= all_tube_lengths

        tube_color_normalized = tube_color[:,:,0:3] / 255.0

        #find smallest coaptation
        plt.scatter(plot_data[:,:,0], plot_data[:,:,1], s = 0.5, c = tube_color_normalized.reshape(-1,3))
        plt.show() 

    def func(self, evt):
        #merge the meshes
        msh = evt.object
        if not msh:
            return
        pt = evt.picked3d
        self.cell_id_array[0,:] = self.cell_id_array[1,:]
        self.cell_id_array[1,:] = pt #idcell
        measurement_point = Sphere(pos = pt, r = 0.3, c= 'white') 
        self.plotter_coaptation.add(measurement_point)
       

        
    ################################################################################

    ##################################   LIP   #####################################
    ################################################################################

    def calculation_LIP(self):
         a = self.annulus_points[40]
         p = self.annulus_points[0]

         a_vis = Sphere(pos=(a), r = 0.7, c = 'pink')
         p_vis = Sphere(pos=(p), r = 0.7, c = 'yellow')

         ap_vis = Line(a, p, lw = 3,c= 'cyan')

         self.lip = a+ 0.1* np.linalg.norm(p-a)* self.normalize(p-a)
         self.lip_vis = Sphere(pos= (self.lip), r = 5, c= 'blue') 

         self.excel_writer.add_value(self.pre_or_intra, 'LIP_position', np.round(self.lip,2))
         
         return self.lip
    
    def calculate_LIP_axis(self):
        axis_end_point = self.lip +15* (self.cutting_plane_coordi[1,:])
        ax_end_lip = Sphere(pos = (axis_end_point), r =1 ,c= 'violet')
        self.plot.add(ax_end_lip)

    def calculate_a_p_diameter(self):
        a = self.annulus_points[40]
        p = self.annulus_points[0]

        a_p_diameter = np.linalg.norm(a-p)
        print('A P Diameter: ', np.round(a_p_diameter, 2), 'mm')
        self.excel_writer.add_value(self.pre_or_intra, 'AP_Diameter', np.round(a_p_diameter, 2))
    ################################################################################

    ###########################  Thin spline model   ###############################
    ################################################################################
    def extend_clip_position(self, clip_position):
        clip_lower_point = clip_position[0,:]
        clip_upper_point = clip_position[1,:]

        length_clip = np.abs(np.linalg.norm(clip_upper_point - clip_lower_point))
        elongation = (self.mpr_height-length_clip)/2

        clip_axis = self.normalize(clip_upper_point-clip_lower_point)

        clip_lower_point_extended = clip_lower_point - clip_axis * elongation
        clip_upper_point_extended  = clip_upper_point + clip_axis * elongation
        clip_position[0,:] = clip_lower_point_extended
        clip_position[1,:] = clip_upper_point_extended
        return clip_position


    def calculate_parallel_intra_model_mprs(self):
        image_dim = np.max([self.volume_intra_op.shape[0], self.volume_intra_op.shape[1], self.volume_intra_op.shape[2]])
        self.extension = np.array([0, image_dim, 0, image_dim])

        clip_position = self.get_clip_mpr_position()
        print('clip_position: ', clip_position)
        clip_position = self.extend_clip_position(clip_position)
        self.calculate_mpr(self.extension, clip_position, False)
        mpr_1 = self.mpr
        t_point_1 = self.rectangle_points[0]
        coordi_1 = self.cutting_plane_coordi
        spline = Spline(self.annulus_points).lw(3)
        intersection_points_mpr_annulus_1 = np.multiply(np.reciprocal(self.slice_thickness),self.convert_list_to_array(self.get_intersection_with_spline(spline, self.mpr_plane_mesh)))
        point_set_1 = clip_position + 1.8 *self.cutting_plane_coordi[2, :] #1.8mm is nearly half of the thickness of the clip (thickness clip = 4mm)
        point_set_2 = clip_position - 1.8 *self.cutting_plane_coordi[2, :]
        self.calculate_mpr(self.extension, point_set_1, False)
        mpr_0 = self.mpr
        t_point_0 = self.rectangle_points[0]
        coordi_0 = self.cutting_plane_coordi
        intersection_points_mpr_annulus_0 = np.multiply(np.reciprocal(self.slice_thickness),self.convert_list_to_array(self.get_intersection_with_spline(spline, self.mpr_plane_mesh)))
        self.calculate_mpr(self.extension, point_set_2, False)
        mpr_2 = self.mpr
        t_point_2 = self.rectangle_points[0]
        coordi_2 = self.cutting_plane_coordi
        intersection_points_mpr_annulus_2 = np.multiply(np.reciprocal(self.slice_thickness),self.convert_list_to_array(self.get_intersection_with_spline(spline, self.mpr_plane_mesh)))

        test_point_1 = Points(intersection_points_mpr_annulus_0, r = 30, c='red')
        test_point_2 = Points(intersection_points_mpr_annulus_1, r = 30, c='blue')
        test_point_3 = Points(intersection_points_mpr_annulus_2, r = 30, c='green')

        saved_clip_position = self.transform_world_to_anatomic(clip_position, self.anatomic_coordi)
        saved_point_set_1 = self.transform_world_to_anatomic(point_set_1, self.anatomic_coordi)
        saved_point_set_2 = self.transform_world_to_anatomic(point_set_2, self.anatomic_coordi)

        self.plot.add(test_point_1, test_point_2, test_point_3)

        return mpr_0.copy(), mpr_1.copy(), mpr_2.copy(), t_point_0.copy(), t_point_1.copy(), t_point_2.copy(), coordi_0.copy(), coordi_1.copy(), coordi_2.copy(), saved_clip_position.copy(), saved_point_set_1.copy(), saved_point_set_2.copy(), intersection_points_mpr_annulus_0.copy(),intersection_points_mpr_annulus_1.copy(),intersection_points_mpr_annulus_2.copy() 
    
    def calculate_pre_model_mprs(self, saved_clip_position, saved_point_set_1, saved_point_set_2):
        clip_position = self.transform_anatomic_to_world(saved_clip_position, self.anatomic_coordi) 
        point_set_1 = self.transform_anatomic_to_world(saved_point_set_1, self.anatomic_coordi)
        point_set_2 = self.transform_anatomic_to_world(saved_point_set_2, self.anatomic_coordi)

        self.calculate_mpr(self.extension, clip_position, False)
        mpr_1 = self.mpr
        t_point_1 = self.rectangle_points[0]
        coordi_1 = self.cutting_plane_coordi
        spline = Spline(self.annulus_points).lw(3)
        intersection_points_mpr_annulus_1 = np.multiply(np.reciprocal(self.slice_thickness),self.convert_list_to_array(self.get_intersection_with_spline(spline, self.mpr_plane_mesh)))
        self.calculate_mpr(self.extension, point_set_1, False)
        mpr_0 = self.mpr
        t_point_0 = self.rectangle_points[0]
        coordi_0 = self.cutting_plane_coordi
        intersection_points_mpr_annulus_0 = np.multiply(np.reciprocal(self.slice_thickness),self.convert_list_to_array(self.get_intersection_with_spline(spline, self.mpr_plane_mesh)))
        self.calculate_mpr(self.extension, point_set_2, False)
        mpr_2 = self.mpr
        t_point_2 = self.rectangle_points[0]
        coordi_2 = self.cutting_plane_coordi
        intersection_points_mpr_annulus_2 = np.multiply(np.reciprocal(self.slice_thickness),self.convert_list_to_array(self.get_intersection_with_spline(spline, self.mpr_plane_mesh)))

        return mpr_0.copy(), mpr_1.copy(), mpr_2.copy(), t_point_0.copy(), t_point_1.copy(), t_point_2.copy(), coordi_0.copy(), coordi_1.copy(), coordi_2.copy(), intersection_points_mpr_annulus_0.copy(),intersection_points_mpr_annulus_1.copy(),intersection_points_mpr_annulus_2.copy() 

    
    
    def calculate_rotated_intra_model_mprs(self):
        image_dim = np.max([self.volume_intra_op.shape[0], self.volume_intra_op.shape[1], self.volume_intra_op.shape[2]])
        self.extension = np.array([0, image_dim, 0, image_dim])
        clip_position = self.get_clip_mpr_position()
        clip_position = self.extend_clip_position(clip_position)
        self.calculate_mpr(self.extension, clip_position, False)
        mpr_1 = self.mpr
        t_point_1 = self.rectangle_points[0]
        coordi_1 = self.cutting_plane_coordi
        spline = Spline(self.annulus_points).lw(3)
        intersection_points_mpr_annulus_1 = np.multiply(np.reciprocal(self.slice_thickness),self.convert_list_to_array(self.get_intersection_with_spline(spline, self.mpr_plane_mesh)))
        
        projected_clip = self.project_point_onto_plane(clip_position[1,:], self.cutting_plane_coordi[1,:], self.lip)
        distance_clip_lip = np.abs(np.linalg.norm(projected_clip-self.lip))
        alpha = 2 * np.arcsin(9 /distance_clip_lip)
        beta = -alpha
        print('LIP:', self.lip)

        point_set_1= np.zeros((3,3))
        point_set_2= np.zeros((3,3))
        for i in range(3):
            point_set_1[i,:]=self.transform_around_LIP_no_scaling(clip_position[i], alpha)

        for i in range(3):
            point_set_2[i,:]=self.transform_around_LIP_no_scaling(clip_position[i], beta)

        
        self.calculate_mpr(self.extension, point_set_1, False)
        mpr_0 = self.mpr
        t_point_0 = self.rectangle_points[0]
        coordi_0 = self.cutting_plane_coordi
        intersection_points_mpr_annulus_0 = np.multiply(np.reciprocal(self.slice_thickness),self.convert_list_to_array(self.get_intersection_with_spline(spline, self.mpr_plane_mesh)))
        self.calculate_mpr(self.extension, point_set_2, False)
        mpr_2 = self.mpr
        t_point_2 = self.rectangle_points[0]
        coordi_2 = self.cutting_plane_coordi
        intersection_points_mpr_annulus_2 = np.multiply(np.reciprocal(self.slice_thickness),self.convert_list_to_array(self.get_intersection_with_spline(spline, self.mpr_plane_mesh)))

        test_point_1 = Points(clip_position, r = 10, c='red')
        test_point_2 = Points(point_set_1, r = 10, c='blue')
        test_point_3 = Points(point_set_2, r = 10, c='green')

        saved_clip_position = self.transform_world_to_anatomic(clip_position, self.anatomic_coordi)
        saved_point_set_1 = self.transform_world_to_anatomic(point_set_1, self.anatomic_coordi)
        saved_point_set_2 = self.transform_world_to_anatomic(point_set_2, self.anatomic_coordi)


        return mpr_0.copy(), mpr_1.copy(), mpr_2.copy(), t_point_0.copy(), t_point_1.copy(), t_point_2.copy(), coordi_0.copy(), coordi_1.copy(), coordi_2.copy(), saved_clip_position.copy(), saved_point_set_1.copy(), saved_point_set_2.copy(), intersection_points_mpr_annulus_0.copy(),intersection_points_mpr_annulus_1.copy(),intersection_points_mpr_annulus_2.copy() 

    ################################################################################

    ###############################   getter   #####################################
    ################################################################################
    
    def get_mpr(self):
        #This is the last mpr that was made, it can be pre intra and/or bent
        return self.mpr
    
    def get_pre_mpr(self):
        return self.mpr_pre
        
    def get_intra_mpr(self):
        return self.mpr_intra

    def get_bent_pre_mpr(self):
        return self.mpr_pre_bent

    def get_bent_intra_mpr(self):
        return self.mpr_intra_bent
    
    def get_mpr_through_clip(self):
        return self.mpr_through_clip
    
    def get_pre_annulus(self):
        return self.annulus_intersection_pre
    
    def get_intra_annulus(self):
        return self.annulus_intersection_intra
    
    def get_bent_pre_annulus(self):
        return self.annulus_intersection_pre_bent
    
    def get_bent_intra_annulus(self):
        return self.annulus_intersection_intra_bent
    
    
    
    def get_all(self):
        volume = self.get_volume()
        mpr_mesh = self.get_mpr_plane_mesh()
        mpr_helper_points = self.get_mpr_mesh_helper_points()
        mpr_coordi = self.get_mpr_coordinate_system()
        annulus_plane = self.get_annulus_plane_mesh()
        annulus_coordi = self.get_annulus_plane_coordinate_system()
        center_gravity = self.get_annulus_plane_center_of_gravity()
        apex = self.get_apex()
        world_coordi = self.get_world_coordi()
        mv_mesh = self.get_mv_mesh()
        ant_mesh = self.get_ant_mesh()
        post_mesh = self.get_post_mesh()
        annulus = self.get_annulus()
        intersection_line = self.get_intersection_line()
        coordi_apex = self.get_coordinate_system_apex()
        bounding_box = self.get_bounding_box()
        axis_apex = self.get_axis_apex_gravity_center()
        aortic_point = self.get_aortic_point()
        clip = self.get_clip()

        item_array = np.concatenate([volume, mpr_mesh, mpr_helper_points, mpr_coordi, annulus_plane, annulus_coordi, center_gravity, apex, world_coordi,
        mv_mesh, ant_mesh, post_mesh, annulus, intersection_line, coordi_apex, bounding_box, axis_apex, aortic_point, clip])
        return item_array
    
    def get_volume(self):
        return [self.vol]
    
    def get_mpr_plane_mesh(self):
        return [self.mpr_plane_mesh]

    def get_mpr_mesh_helper_points(self):
        return [self.p0_vis, self.p1_vis, self.p2_vis, self.p3_vis]
    
    def get_mpr_coordinate_system(self):
        return [self.e1_vis, self.e2_vis, self.e3_vis]
    
    def get_annulus_plane_mesh(self):
        return [self.best_fit_plane]
    
    def get_annulus_plane_coordinate_system(self):
        return [self.bf_e1_vis, self.bf_e2_vis, self.bf_e3_vis]
    
    def get_annulus_plane_center_of_gravity(self):
        return [self.gravity_center_annulus_vis]
    
    def get_mv_mesh(self):
        return [self.mv_model_mesh]
    
    def get_ant_mesh(self):
        return [self.ant_model_mesh]
    
    def get_post_mesh(self):
        return [self.post_model_mesh]
    
    def get_apex(self):
        return [self.apex_mesh]
    
    def get_world_coordi(self):
        return [self.e1_world_vis, self.e2_world_vis, self.e3_world_vis]
    
    def get_annulus(self):
        return [self.annulus]
    
    def get_intersection_line(self):
        return [self.clip_intersection]
    
    def get_coordinate_system_apex(self):
        return [self.bf_e1_vis_apex, self.bf_e2_vis_apex, self.bf_e3_vis_apex]
    
    def get_bounding_box(self):
        return[self.bounding_box]
    
    def get_axis_apex_gravity_center(self):
        return [self.axis_apex_grav_center]
    
    def get_aortic_point(self):
        return [self.aortic_point_proj_vis]
    
    def get_croped_volume(self):
        return [self.croped_vol]
    
    def get_clip(self):
        return [self.clip_vis]
  

    
    def get_all_visibility(self):
        item_array = self.get_all()

        self.all_visible = not self.all_visible
        return item_array, not self.all_visible
    
    def get_volume_and_visibility(self):
        item = self.get_volume()
        self.vol_visible = not self.vol_visible
        return item, not self.vol_visible

    def get_mpr_plane_mesh_and_visibility(self):
        item = self.get_mpr_plane_mesh()
        self.mpr_plane_mesh_visible = not self.mpr_plane_mesh_visible
        return item, not self.mpr_plane_mesh_visible
    
    def get_mpr_mesh_helper_points_and_visibility(self):
        item = self.get_mpr_mesh_helper_points()
        self.pts_vis_visible = not self.pts_vis_visible
        return item, not self.pts_vis_visible
    
    def get_mpr_coordinate_system_and_visibility(self):
        item = self.get_mpr_coordinate_system()
        self.e_coordi_vis_visible = not self.e_coordi_vis_visible
        return item, not self.e_coordi_vis_visible
    
    def get_annulus_plane_mesh_and_visibility(self):
        item = self.get_annulus_plane_mesh()
        self.best_fit_plane_visible = not self.best_fit_plane_visible
        return item, not self.best_fit_plane_visible
    
    def get_annulus_plane_coordinate_system_and_visibility(self):
        item = self.get_annulus_plane_coordinate_system()
        self.bf_coordi_vis_visible = not self.bf_coordi_vis_visible
        return item, not self.bf_coordi_vis_visible
    
    def get_annulus_plane_center_of_gravity_and_visibility(self):
        item = self.get_annulus_plane_center_of_gravity()
        self.gravity_center_annulus_vis_visible = not self.gravity_center_annulus_vis_visible
        return item, not self.gravity_center_annulus_vis_visible
    
    def get_apex_and_visibility(self):
        item = self.get_apex()
        self.apex_vis_visible = not self.apex_vis_visible
        return item, not self.apex_vis_visible
    
    def get_world_coordi_and_visibility(self):
        item = self.get_world_coordi()
        self.world_coordi_visible = not self.world_coordi_visible
        return item, not self.world_coordi_visible
    
    def get_mv_mesh_and_visibility(self):
        item = self.get_mv_mesh()
        self.mv_model_mesh_visible = not self.mv_model_mesh_visible
        return item, not self.mv_model_mesh_visible
    
    def get_ant_mesh_and_visibility(self):
        item = self.get_ant_mesh()
        self.ant_model_mesh_visible = not self.ant_model_mesh_visible
        return item, not self.ant_model_mesh_visible
    
    def get_post_mesh_and_visibility(self):
        item = self.get_post_mesh()
        self.post_model_mesh_visible = not self.post_model_mesh_visible
        return item, not self.post_model_mesh_visible
    
    def get_annulus_and_visibility(self):
        item = self.get_annulus()
        self.annulus_visible = not self.annulus_visible
        return item, not self.annulus_visible
    
    def get_intersection_line_and_visibility(self):
        item = self.get_intersection_line()
        self.intersection_line_visible = not self.intersection_line_visible
        return item, not self.intersection_line_visible
    
    def get_coordinate_system_apex_and_visibility(self):
        item = self.get_coordinate_system_apex()
        self.apex_coordi_visible = not self.apex_coordi_visible
        return item, not self.apex_coordi_visible
    
    def get_bounding_box_and_visibility(self):
        item = self.get_bounding_box()
        self.bounding_box_visible = not self.bounding_box_visible
        return item, not self.bounding_box_visible
    
    def get_axis_apex_gravity_center_and_visibility(self):
        item = self.get_axis_apex_gravity_center()
        self.axis_apex_gravity_center_visible = not self.axis_apex_gravity_center_visible
        return item, not self.axis_apex_gravity_center_visible
    
    def get_aortic_point_and_visibility(self):
        item = self.get_aortic_point()
        self.aortic_point_visibile = not self.aortic_point_visibile
        return item, not self.aortic_point_visibile
    
    def get_croped_volume_and_visibility(self):
        item = self.get_croped_volume()
        self.croped_volume_visible = not self.croped_volume_visible
        return item, not self.croped_volume_visible
    
    def get_clip_and_visibility(self):
        item = self.get_clip()
        self.clip_visible = not self.clip_visible
        return item, not self.clip_visible

    def get_model_None_state(self):
        none_state = [model is None for model in (self.mv_model, self.ant_model, self.post_model, self.spline_model)]
        return none_state

    ################################################################################

    

    ################################   saving   ####################################
    ################################################################################

    def save_single_mpr(self):
        if self.mpr_mode_pre == True:
            mpr = self.mpr_pre
        elif self.mpr_mode_pre == False:
            mpr = self.mpr_intra

        self.save_mpr(mpr)

    def save_mpr(self, mpr_to_save, nr =None):
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y_%m_%d__%H_%M_%S")
        if nr is not None:
            file_name = 'mpr_'+ formatted_datetime + '_' + str(nr)+'.png'
        else:
            file_name = 'mpr_'+ formatted_datetime + '.png'
        path_folder = #
        path = path_folder + file_name
        image = Image.fromarray(mpr_to_save)
        image = image.convert('RGB')
        image.save(path)
        print('MPR was saved to ', path)

        #save as float
        '''
        file_name = 'mpr_'+ formatted_datetime + '_float.npy''
        path_folder = #
        path = path_folder + file_name
        np.save(path, self.mpr)
        print('MPR was saved to ', path)'''

    def save_mpr_set(self):
        for i in range(self.all_slices.shape[2]):
            self.save_mpr(self.all_slices[:,:,i],i)

    ################################################################################

    
    
    
     
        






