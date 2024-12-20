from PySide6.QtWidgets import (QTabWidget)
from mpr2D_tab import MPR_2D_Tab
from model3D_tab import Model_3D_Tab 
from mpr_slicer import MPR_Slicer
from intervention_helpers_tab import Intervention_Helpers_Tab
from clipped_spline_length_tab import Clipped_Spline_Length_Tab
from thin_spline_model import Thin_spline_model

class View_tabs(QTabWidget):
    def __init__(self, volume_pre_op, volume_intra_op, mv_model_pre, ant_model_pre, post_model_pre, mv_model_intra, ant_model_intra, post_model_intra, slice_thickness_pre, slice_thickness_intra, spline_model_pre, spline_model_intra, apex_pre, apex_intra, clip_start_pre, clip_start_intra, clip_end_pre, clip_end_intra, annulus_pre_data, annulus_intra_data, excel_writer, dataloader):
        super().__init__()

        self.mpr_slicer = MPR_Slicer(volume_pre_op, volume_intra_op, spline_model_pre, spline_model_intra, annulus_pre_data, annulus_intra_data, excel_writer, dataloader)
        self.tab_3D = Model_3D_Tab(self.mpr_slicer, volume_pre_op, volume_intra_op, mv_model_pre, ant_model_pre, post_model_pre, mv_model_intra, ant_model_intra, post_model_intra, slice_thickness_pre, slice_thickness_intra, apex_pre, apex_intra, excel_writer, dataloader)
        self.tab_2D = MPR_2D_Tab(self.mpr_slicer, False, excel_writer)
        self.tab_2D_bent = MPR_2D_Tab(self.mpr_slicer, True, excel_writer)
        self.tab_intervention_helpers = Intervention_Helpers_Tab(self.mpr_slicer)
        self.tab_clipped_spline_length = Clipped_Spline_Length_Tab(self.mpr_slicer, self.tab_2D, excel_writer)
        self.tab_thin_spline_model = Thin_spline_model(self.mpr_slicer, excel_writer, dataloader)



        self.tabs = {"volume": self.tab_3D, "mpr" : self.tab_2D,  "bent": self.tab_2D_bent, "helpers": self.tab_intervention_helpers}
        self.addTab(self.tab_3D, "3D Volume")
        self.addTab(self.tab_2D, "2D MPR through clip")
        self.addTab(self.tab_2D_bent, "2D Test for bent leaflet")
        self.addTab(self.tab_clipped_spline_length, "Clipped spline length")
        self.addTab(self.tab_intervention_helpers, "Intervention helpers")
        self.addTab(self.tab_thin_spline_model, "Thin spline model")
        
        

