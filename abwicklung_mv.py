from dataloader import Dataloader
from vedo import *
import numpy as np
import matplotlib.pyplot as plt 

plotter = Plotter()

dataloader= Dataloader()

def calculate_spline_length(spline):
    dist = np.zeros(40)
    for i in range(40-1):
        dist[i+1] = np.linalg.norm(spline[i,:]-spline[i+1,:])
    total_length= np.cumsum(dist)
    return total_length


def create_unfolding():
    spline_model, u_v_data_split = dataloader.read_spline_model()
    all_tube_lengths = np.zeros((79, 40))
    for i in range(79):
        
        single_spline = Spline(spline_model[i,:,:])
        all_tube_lengths[i,:] = calculate_spline_length(spline_model[i,:,:])
        plotter.add(single_spline)

    plot_data= np.zeros((79,40,2))
    plot_data[:,:,0]= u_v_data_split[:,:,1]
    plot_data[:,:,1]= all_tube_lengths

    plt.scatter(plot_data[:,:,0], plot_data[:,:,1], s = 0.5, )
    plt.show() 