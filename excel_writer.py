from pandas import DataFrame
import pandas as pd
from datetime import datetime

class Excel_Writer():
    
    def __init__(self, observer_name, mesh_name):
        self.file_name = self.create_unique_file_name(observer_name, mesh_name)
        self.make_initial_excel_sheet(self.file_name)

    def create_unique_file_name(self, observer_name, mesh_name):
        path = #
        time_step = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
        unique_file_name = path + observer_name + '_' + mesh_name + '_' + time_step + '.xlsx'
        
        return unique_file_name

    def make_initial_excel_sheet(self, file_name):
        # Define the categorical names
        categories = ['observer', 'round', 'mesh_name', 'leaflet_length_ant_intersection', 'leaflet_length_post_intersection',
                      'leaflet_length_ant_spline', 'leaflet_length_post_spline',
                      'leaflet_length_clipped_complete_ant', 'leaflet_length_clipped_partial_ant', 'leaflet_length_clipped_without_ant','leaflet_length_clipped_neighbor_ant',
                      'leaflet_length_clipped_complete_post', 'leaflet_length_clipped_partial_post', 'leaflet_length_clipped_without_post','leaflet_length_clipped_neighbor_post',
                      'coaptation_height', 'coaptation_width', 'coaptation_width_closure_line', 
                      'MVOA_cl', 'MVOA_annulus', 'MVOA_main', 'angle_anatomic_coordi', 
                      'percent_AP_a_n_0', 'percent_AP_a_n_1', 'percent_AP_a_0', 'percent_AP_a_1', 'percent_AP_n_0', 'percent_AP_n_1',
                      'bent_ant_detected', 'bent_ant_reality', 'bent_post_detected', 'bent_post_reality','leaflet_area', 
                      'AP_Diameter', 'Clip_position', 'Clip_orientation', 'MV_area', 'LIP_position',
                      'mv_model_complete', 'ant_model_complete', 'post_model_complete', 'dat_data', 'raw_data', 'spline_model_complete', 'apex', 
                      'mv_model_partial', 'ant_model_partial', 'post_model_partial','spline_model_partial',
                      'length_complete_ant', 'length_complete_post',
                      'length_partial_ant','length_partial_post',
                      'length_without_ant','length_without_post',
                      'length_neighbor_ant','length_neighbor_post',
                      'length_bent_ant','length_bent_post',
                      'number_waypoints_ant_complete', 'number_waypoints_post_complete',
                      'number_waypoints_ant_partial', 'number_waypoints_post_partial',
                      'number_waypoints_ant_without', 'number_waypoints_post_without',
                      'number_waypoints_ant_neighbor', 'number_waypoints_post_neighbor',
                      'number_waypoints_ant_bent', 'number_waypoints_post_bent',
                      'selection_time_ant_complete', 'selection_time_post_complete',
                      'selection_time_ant_partial', 'selection_time_post_partial',
                      'selection_time_ant_without', 'selection_time_post_without',
                      'selection_time_ant_neighbor', 'selection_time_post_neighbor',
                      'selection_time_ant_bent', 'selection_time_post_bent']


        categories_df = pd.DataFrame(categories, columns=['Category'])

        # Create a DataFrame for the header row
        header_df = pd.DataFrame([['Category', 'pre', 'intra']], columns=['Category', 'pre', 'intra'])

        # Concatenate the header and categories DataFrames
        self.df = pd.concat([header_df, categories_df], ignore_index=True)


    def add_value(self, pre_or_intra, category, value):
        index = self.df[self.df['Category'] == category].index[0]

        # Add the entry in the 'second' column for the 'volume' row
        self.df.at[index, pre_or_intra] = value  # Replace 'your_entry' with the desired value

        # Define the name of the Excel file
        excel_file_name = self.file_name

    def save_annulus_intersection_points(self, pre_or_intra, value, number, normal):
        if normal == True: 
            category = 'annulus_intersection_normal'
        elif normal == False:
            category = 'annulus_intersection_apex'

        category = category + '_' + str(number)
    
        self.add_value(pre_or_intra, category, value)

    def save_excel_sheet(self):
        self.df.to_excel(self.file_name, index=False, header=False)

        print(f'Excel file "{self.file_name}" created successfully with the categories.')


my_excel_writer = Excel_Writer('Bella', 'No_mesh')


    
    
