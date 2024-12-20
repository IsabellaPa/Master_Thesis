from vedo import Mesh, Plotter, show, Tube, Spline, Volume
import numpy as np
import re
import vedo.vtkclasses as vtk
from vedo import utils

file_path = r'C:\\Users\\ipatzke\\OneDrive - Philips\\Documents\\Masterarbeit\\MasterMission\\Resourcen\\Tissue_6\\mv.inp'
def load_inp_file(file_path):
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
def get_volume_dimensions(file_path):  
        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(r'Resolution:\s+(\d+)\s+(\d+)\s+(\d+)', line)
                
                if match:
                    dimensions = [int(match.group(i)) for i in range(1, 4)]
        return dimensions

def open_volume():
        file =  r'C:\\Users\\ipatzke\\OneDrive - Philips\\Documents\\Masterarbeit\\MasterMission\\Resourcen\\Tissue_6\\1_Tissue_006.raw'
        dat_file = r'C:\\Users\\ipatzke\\OneDrive - Philips\\Documents\\Masterarbeit\\MasterMission\\Resourcen\\Tissue_6\\1_Tissue_006.dat'
        Dim_size = get_volume_dimensions(dat_file)

        f = open(file,'rb') 
        img_arr=np.fromfile(f,dtype=np.uint8)

        img_arr = img_arr[0:Dim_size[0]*Dim_size[1]*Dim_size[2]] 
        img_arr = img_arr.reshape(Dim_size[2],Dim_size[1],Dim_size[0])

        volume =np.transpose(img_arr)
        return volume


def crop_along_plane(volume_points, plane_normal, plane_point):
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

    #[coordinate_x, coordinate_y, coordinate_z, color_value, distance_to_plane]
    volume_with_distance = np.zeros((volume_points.shape[0]* volume_points.shape[1]* volume_points.shape[2], 5))
    volume_with_distance[:,0:3] = coordinates_in_volume
    volume_with_distance[:, 3] = np.reshape(volume_points,(-1))
    volume_with_distance[:, 4] = distance_from_plane

    negative_rows = volume_with_distance[:, 4] < 0
    volume_with_distance[negative_rows, 3] = 0

    croped_volume = np.reshape(volume_with_distance[:,3],(volume_points.shape[0], volume_points.shape[1], volume_points.shape[2]))

    return croped_volume



'''mv_vertices, mv_faces = load_inp_file(file_path)
mv_mesh = Mesh([mv_vertices, mv_faces])
volume = open_volume()
vol = Volume(volume)
croped_volume = crop_along_plane(volume, [0,5,1], [100,100,100])
vol.spacing(s = (0.5, 0.5,0.5))
new_volume = Volume(croped_volume)
show(new_volume)'''