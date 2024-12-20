from PySide6.QtCore import QTimer,QDateTime, Qt, Slot,QPointF, Signal
from PySide6.QtGui import QAction, QPainter,QImage, QPixmap, QPen, QIntValidator, QColor, QKeySequence, QPolygon, QPolygonF, QBrush, QMouseEvent, QFont, QCursor
from PySide6.QtWidgets import (QApplication, QLabel,QMainWindow, QPushButton, QWidget, QListWidget,
                               QLineEdit,QFileDialog,QVBoxLayout, QHBoxLayout,QDialogButtonBox,
                               QGridLayout, QCheckBox, QMessageBox, QSizePolicy, QSlider, QGraphicsView, 
                               QGraphicsScene, QStyle,QSpacerItem, QRadioButton, QTabWidget)
import numpy as np

class Intervention_Helpers_Tab(QWidget):
    def __init__(self, mpr_slicer):
        super().__init__()

        self.mpr_slicer = mpr_slicer
        self.normal_for_MVOA = 'cl'
        self.single_MVOA = True
        self.dist_MVOA = 0

        self.mpr_slicer.intervention_helpers = self


        button_layout = QVBoxLayout()

        self.calculate_MVOA_button = QPushButton("Calculate MVOA")
        self.calculate_MVOA_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.calculate_MVOA_button.clicked.connect(self.calculate_MVOA_button_was_pressed)
        self.single_vs_set_MVOA_button = QPushButton("MVOA set")
        self.single_vs_set_MVOA_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.single_vs_set_MVOA_button.clicked.connect(self.single_vs_set_MVOA_button_was_pressed)
        self.calculate_MV_area_button = QPushButton("Calculate MV area")
        self.calculate_MV_area_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.calculate_MV_area_button.clicked.connect(self.calculate_MV_area_button_was_pressed)
        self.show_closure_line_heigth_button = QPushButton("Show closureline height")
        self.show_closure_line_heigth_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.show_closure_line_heigth_button.clicked.connect(self.show_closure_line_heigth_button_was_pressed)
        self.show_coaptation_area_button = QPushButton("Show coaptation area")
        self.show_coaptation_area_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.show_coaptation_area_button.clicked.connect(self.show_coaptation_area_button_was_pressed)
        self.calculate_leaflet_area_button = QPushButton("Calculate leaflet area")
        self.calculate_leaflet_area_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.calculate_leaflet_area_button.clicked.connect(self.calculate_leaflet_area_button_was_pressed)
        self.calculate_a_p_diameter_button = QPushButton("Calculate A P diameter")
        self.calculate_a_p_diameter_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.calculate_a_p_diameter_button.clicked.connect(self.calculate_a_p_diameter_button_was_pressed)
        
        

        self.measure_coaptation_height_button = QPushButton("Measure coaptation height (Calculate coaptation area first)")
        self.measure_coaptation_height_button.setEnabled(False)
        self.measure_coaptation_height_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.measure_coaptation_height_button.clicked.connect(self.measure_coaptation_height_button_was_pressed)
        self.measure_coaptation_width_button = QPushButton("Measure coaptation width (Calculate coaptation area first)")
        self.measure_coaptation_width_button.setEnabled(False)
        self.measure_coaptation_width_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.measure_coaptation_width_button.clicked.connect(self.measure_coaptation_width_button_was_pressed)
        self.measure_coaptation_width_curvy_button = QPushButton("Measure coaptation width - closure line(Calculate coaptation area first)")
        self.measure_coaptation_width_curvy_button.setEnabled(False)
        self.measure_coaptation_width_curvy_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.measure_coaptation_width_curvy_button.clicked.connect(self.measure_coaptation_width_curvy_button_was_pressed)


        self.show_unfolding_button = QPushButton("Show unfolding (Calculate coaptation area first)")
        self.show_unfolding_button.setEnabled(False)
        self.show_unfolding_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.show_unfolding_button.clicked.connect(self.show_unfolding_button_was_pressed)
      

        self.normal_MVOA_cl_radio_button = QRadioButton("normal of closure line")
        self.normal_MVOA_cl_radio_button.setChecked(True)
        self.normal_MVOA_cl_radio_button.toggled.connect(self.normal_MVOA_cl_selected)
        self.normal_MVOA_annulus_radio_button = QRadioButton("normal of annulus")
        self.normal_MVOA_annulus_radio_button.setChecked(False)
        self.normal_MVOA_annulus_radio_button.toggled.connect(self.normal_MVOA_annulus_selected)
        self.normal_MVOA_main_axis_radio_button = QRadioButton("normal main axis")
        self.normal_MVOA_main_axis_radio_button.setChecked(False)
        self.normal_MVOA_main_axis_radio_button.toggled.connect(self.normal_MVOA_main_axis_selected)
        self.normal_MVOA_costum_radio_button = QRadioButton("costum normal")
        self.normal_MVOA_costum_radio_button.setChecked(False)
        self.normal_MVOA_costum_radio_button.toggled.connect(self.normal_MVOA_costum_selected)

        self.calculate_clip_orientation_button = QPushButton("Calculate the clip orientaion")
        self.calculate_clip_orientation_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.calculate_clip_orientation_button.clicked.connect(self.calculate_clip_orientation_button_was_pressed)
        self.multiple_clip_distance_button = QPushButton("Calculate distance of multiple clips")
        self.multiple_clip_distance_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.multiple_clip_distance_button.clicked.connect(self.multiple_clip_distance_button_was_pressed)

        

        self.normal_MVOA_x = QLineEdit()
        self.normal_MVOA_x.setText(str(''))
        self.normal_MVOA_x.setValidator(QIntValidator())
        self.normal_MVOA_x.setMaxLength(3)
        self.normal_MVOA_x.setAlignment(Qt.AlignRight)
        self.normal_MVOA_x_label = QLabel("x: ")

        self.normal_MVOA_y = QLineEdit()
        self.normal_MVOA_y.setText(str(''))
        self.normal_MVOA_y.setValidator(QIntValidator())
        self.normal_MVOA_y.setMaxLength(3)
        self.normal_MVOA_y.setAlignment(Qt.AlignRight)
        self.normal_MVOA_y_label = QLabel("y: ")

        self.normal_MVOA_z = QLineEdit()
        self.normal_MVOA_z.setText(str(''))
        self.normal_MVOA_z.setValidator(QIntValidator())
        self.normal_MVOA_z.setMaxLength(3)
        self.normal_MVOA_z.setAlignment(Qt.AlignRight)
        self.normal_MVOA_z_label = QLabel("z: ")

        self.number_of_MVOA_slices = QLineEdit()
        self.number_of_MVOA_slices.setText(str('5'))
        self.number_of_MVOA_slices.setValidator(QIntValidator())
        self.number_of_MVOA_slices.setMaxLength(2)
        self.number_of_MVOA_slices.setAlignment(Qt.AlignRight)
        self.number_of_MVOA_slices_label = QLabel("number of slices: ")
        

        self.slider_dist_MVOA = QSlider(Qt.Horizontal, self)
        self.slider_dist_MVOA.setMinimum(0)
        self.slider_dist_MVOA.setMaximum(10)
        self.slider_dist_MVOA.setSingleStep(1)
        self.slider_dist_MVOA.setValue(10)
        self.slider_dist_MVOA_label = QLabel("distance: ")
        self.slider_dist_MVOA.sliderMoved.connect(self.slider_dist_MVOA_was_moved)


        self.number_of_clip = QLineEdit()
        self.number_of_clip.setText(str('0'))
        self.number_of_clip.setValidator(QIntValidator())
        self.number_of_clip.setMaxLength(2)
        self.number_of_clip.setAlignment(Qt.AlignRight)
        self.number_of_clip_label = QLabel("number of clip: ")

        self.number_of_clip_1 = QLineEdit()
        self.number_of_clip_1.setText(str('0'))
        self.number_of_clip_1.setValidator(QIntValidator())
        self.number_of_clip_1.setMaxLength(2)
        self.number_of_clip_1.setAlignment(Qt.AlignRight)
        self.number_of_clip_1_label = QLabel("number of first clip: ")

        self.number_of_clip_2 = QLineEdit()
        self.number_of_clip_2.setText(str('1'))
        self.number_of_clip_2.setValidator(QIntValidator())
        self.number_of_clip_2.setMaxLength(2)
        self.number_of_clip_2.setAlignment(Qt.AlignRight)
        self.number_of_clip_2_label = QLabel("number of second clip: ")
        

        #button_layout.addWidget(self.show_closure_line_heigth_button)
        button_layout.addWidget(self.show_coaptation_area_button)
        button_layout.addWidget(self.measure_coaptation_height_button)
        button_layout.addWidget(self.measure_coaptation_width_button)
        button_layout.addWidget(self.measure_coaptation_width_curvy_button)
        button_layout.addWidget(self.show_unfolding_button)
        button_layout.addWidget(self.calculate_MV_area_button)
        button_layout.addWidget(self.calculate_leaflet_area_button)
        button_layout.addWidget(self.calculate_a_p_diameter_button)
        button_layout.addWidget(self.calculate_MVOA_button)

        MVOA_layout = QGridLayout()
        MVOA_layout.addWidget(self.normal_MVOA_cl_radio_button, 1,0)
        MVOA_layout.addWidget(self.normal_MVOA_annulus_radio_button, 1,1)
        MVOA_layout.addWidget(self.normal_MVOA_main_axis_radio_button, 1,2)
        MVOA_layout.addWidget(self.normal_MVOA_costum_radio_button, 2,0)
        MVOA_layout.addWidget(self.normal_MVOA_x_label,2,1)
        MVOA_layout.addWidget(self.normal_MVOA_x,2,2)
        MVOA_layout.addWidget(self.normal_MVOA_y_label,2,3)
        MVOA_layout.addWidget(self.normal_MVOA_y,2,4)
        MVOA_layout.addWidget(self.normal_MVOA_z_label,2,5)
        MVOA_layout.addWidget(self.normal_MVOA_z,2,6)
        MVOA_layout.addWidget(self.single_vs_set_MVOA_button,1,3)
        MVOA_layout.addWidget(self.number_of_MVOA_slices_label,1,4)
        MVOA_layout.addWidget(self.number_of_MVOA_slices,1,5)
        MVOA_layout.addWidget(self.slider_dist_MVOA_label,3,0)
        MVOA_layout.addWidget(self.slider_dist_MVOA,3,1,  1,6)

        clip_layout = QGridLayout()
        clip_layout.addWidget(self.calculate_clip_orientation_button,0,0)
        clip_layout.addWidget(self.multiple_clip_distance_button,1,0)
        clip_layout.addWidget(self.number_of_clip_label,0,1)
        clip_layout.addWidget(self.number_of_clip,0,2)
        clip_layout.addWidget(self.number_of_clip_1_label,1,1)
        clip_layout.addWidget(self.number_of_clip_1,1,2)
        clip_layout.addWidget(self.number_of_clip_2_label,1,3)
        clip_layout.addWidget(self.number_of_clip_2,1,4)

       

        window_layout = QGridLayout()
        window_layout.addLayout(button_layout, 0,0)
        window_layout.addLayout(MVOA_layout, 1,0)
        window_layout.addLayout(clip_layout, 2,0)
        
        self.setLayout(window_layout)

        self.disable_enable_buttons()


    def calculate_MVOA_button_was_pressed(self):
        if self.normal_for_MVOA == 'cl':
            normal = self.mpr_slicer.cl_bf_e3
            main_axis = False
            mode = 'MVOA_cl'
        elif self.normal_for_MVOA == 'annulus':
            normal = self.mpr_slicer.bf_e3
            main_axis = False
            mode = 'MVOA_annulus'
        elif self.normal_for_MVOA == 'main axis':
            normal = self.mpr_slicer.cl_bf_e3 
            main_axis = True
            mode = 'MVOA_main'
        elif self.normal_for_MVOA == 'custom':
            normal = np.array([int(self.normal_MVOA_x.text()), int(self.normal_MVOA_y.text()), int(self.normal_MVOA_z.text())])
            main_axis = False
            mode = 'None'
        number_of_slices = int(self.number_of_MVOA_slices.text())
        self.mpr_slicer.calculate_MVOA(normal, self.single_MVOA, (self.dist_MVOA/10), number_of_slices, mode, main_axis)

    def calculate_MV_area_button_was_pressed(self):
        self.mpr_slicer.calculate_MV_area()

    def show_closure_line_heigth_button_was_pressed(self):
        pass

    def show_coaptation_area_button_was_pressed(self):
        print('Coaptation area is calculated...')
        self.show_unfolding_button.setText('Show unfolding')
        self.show_unfolding_button.setEnabled(True)
        self.measure_coaptation_height_button.setText('Measure coaptation height')
        self.measure_coaptation_height_button.setEnabled(True)
        self.measure_coaptation_width_button.setText('Measure coaptation width - straight')
        self.measure_coaptation_width_button.setEnabled(True)
        self.measure_coaptation_width_curvy_button.setText('Measure coaptation width - closure line')
        self.measure_coaptation_width_curvy_button.setEnabled(True)
        self.mpr_slicer.color_mesh_with_distance_values()

    def calculate_leaflet_area_button_was_pressed(self):
        self.mpr_slicer.leaflet_area()

    def calculate_a_p_diameter_button_was_pressed(self):
        self.mpr_slicer.calculate_a_p_diameter()

    def measure_coaptation_height_button_was_pressed(self):
        self.mpr_slicer.measure_coaptation_height()
    
    def measure_coaptation_width_button_was_pressed(self):
        self.mpr_slicer.measure_coaptation_width()

    def measure_coaptation_width_curvy_button_was_pressed(self):
        self.mpr_slicer.measure_coaptation_width_curvy()

    def calculate_clip_orientation_button_was_pressed(self):
        self.mpr_slicer.calculate_clip_orientation(int(self.number_of_clip.text()))

    def multiple_clip_distance_button_was_pressed(self):
        self.mpr_slicer.calculate_multiple_clip_distance(int(self.number_of_clip_1.text()), int(self.number_of_clip_2.text()))

    def show_unfolding_button_was_pressed(self):
        self.mpr_slicer.create_unfolding()

    def normal_MVOA_cl_selected(self):
        self.normal_for_MVOA = 'cl'

    def normal_MVOA_annulus_selected(self):
        self.normal_for_MVOA = 'annulus'

    def normal_MVOA_main_axis_selected(self):
        self.normal_for_MVOA = 'main axis'

    def normal_MVOA_costum_selected(self):
        self.normal_for_MVOA = 'custom'

    def single_vs_set_MVOA_button_was_pressed(self):
        self.single_MVOA = not self.single_MVOA
        if self.single_MVOA == True:
            self.single_vs_set_MVOA_button.setText('MVOA set')
        elif self.single_MVOA == False:
            self.single_vs_set_MVOA_button.setText('MVOA single')

        
    def slider_dist_MVOA_was_moved(self):
        self.dist_MVOA = self.slider_dist_MVOA.value()

    def disable_enable_buttons(self):
        mv_none, ant_none, post_none, spline_none= self.mpr_slicer.get_model_None_state()

        if mv_none or ant_none or post_none:
            self.show_coaptation_area_button.setEnabled(False)
        if not mv_none or not ant_none or not post_none:
            self.show_coaptation_area_button.setEnabled(True)

        if mv_none:
            self.calculate_a_p_diameter_button.setEnabled(False)
            self.calculate_leaflet_area_button.setEnabled(False)
            self.calculate_MV_area_button.setEnabled(False)
            self.calculate_MVOA_button.setEnabled(False)
        else:
            self.calculate_a_p_diameter_button.setEnabled(True)
            self.calculate_leaflet_area_button.setEnabled(True)
            self.calculate_MV_area_button.setEnabled(True)
            self.calculate_MVOA_button.setEnabled(True)
            
        if spline_none:
            self.show_closure_line_heigth_button.setEnabled(False)
            self.calculate_clip_orientation_button.setEnabled(False)
            self.calculate_MVOA_button.setEnabled(False)
        else:
            self.show_closure_line_heigth_button.setEnabled(True)
            self.calculate_clip_orientation_button.setEnabled(True)
            self.calculate_MVOA_button.setEnabled(True)


  