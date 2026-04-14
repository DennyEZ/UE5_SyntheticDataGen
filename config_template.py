# =============================================================================
# UE5 Synthetic Data Generator — Configuration Template
# =============================================================================
# Copy this file to 'config.py' and edit the values for your environment.
# config.py is gitignored — your local settings won't be committed.
# =============================================================================

# =============================================================================
# SHARED — used by all generator scripts
# =============================================================================

# Scene Tags (actor tags in UE5)
TARGET_TAG = "TrainObject"
CAMERA_TAG = "AUV_Camera"
IGNORE_TAG = "IgnoreObject"
HIDE_IN_NEGATIVE_TAG = "HideInNegative"

# Pool / Environment Bounds (cm)
POOL_BOUNDS = {
    "x_min": -1776.0, "x_max": 989.0,
    "y_min": -3992.0, "y_max": 690.0,
    "z_min": -1841.0, "z_max": -1360.0
}

# Camera Intrinsics
SENSOR_WIDTH_MM = 36.0
SENSOR_HEIGHT_MM = 20.25      # 16:9 aspect ratio (36 / 1.777...)
FOCAL_LENGTH_MM = 30.0

# Default Render Resolution
RESOLUTION_X = 1920
RESOLUTION_Y = 1080

# Default Render Settings
WARMUP_FRAMES = 64
SPATIAL_SAMPLES = 1
TEMPORAL_SAMPLES = 1           # Keep at 1 to avoid ghosting

# =============================================================================
# YOLO V3 — Per-Object Generation  (generate_yolo_v3.py)
# =============================================================================
# See object_registry.py for all valid object/group names.

# What to generate. Options:
#   ["all"]                           — every object in the registry
#   ["cam_front"]                     — all objects in the cam_front group
#   ["cam_bottom"]                    — all objects in the cam_bottom group
#   ["gate_sawfish", "octagon"]       — specific objects by name
#   ["cam_front", "bottle"]           — mix groups and individual objects
YOLO_V3_GENERATE = ["all"]

# Root output folder (camera_group/object sub-folders created inside)
YOLO_V3_OUTPUT_ROOT = "C:/UE5_YOLO_Data_V3/"

# Sequencer path prefix (per-object sequences created as children)
YOLO_V3_SEQUENCE_PREFIX = "/Game/Generated/YOLOV3"

# Generation mode: "detect" or "segment"
YOLO_V3_MODE = "detect"

# Detect-mode filter for very small projected boxes at low resolutions.
# Boxes smaller than these pixel thresholds are skipped.
YOLO_V3_MIN_BBOX_WIDTH_PX = 3
YOLO_V3_MIN_BBOX_HEIGHT_PX = 50

# Detect-mode occlusion handling:
#   "off"    = keep geometric boxes only
#   "drop"   = drop farther boxes when projected overlap suggests occlusion
#   "refine" = run targeted mask refinement only for flagged farther objects
YOLO_V3_OCCLUSION_MODE = "off"
YOLO_V3_OCCLUSION_OVERLAP_RATIO = 0.25
YOLO_V3_OCCLUSION_DEPTH_MARGIN_CM = 10.0

# =============================================================================
# YOLO Detection  (generate_yolo_data.py)
# =============================================================================

YOLO_OUTPUT_FOLDER = "C:/UE5_YOLO_Data/"
YOLO_SEQUENCE_PATH = "/Game/Generated/YOLOSequence"
YOLO_SAMPLES_PER_OBJECT = 10
YOLO_VAL_SPLIT_RATIO = 0.2
YOLO_NEGATIVE_SAMPLE_RATIO = 0.1

YOLO_ENABLE_CAM_JITTER = True
YOLO_CAM_JITTER_MAX_PITCH = 5.0
YOLO_CAM_JITTER_MAX_YAW = 5.0

YOLO_MIN_DISTANCE = 100.0     # cm
YOLO_MAX_DISTANCE = 400.0     # cm

YOLO_USE_MASK_BBOX = False
YOLO_MIN_CONTOUR_AREA = 10
YOLO_MIN_VISIBLE_PIXELS = 15

YOLO_RANDOMIZE_OBJECTS = True
YOLO_OBJECT_XY_RANGE_X = 18.0     # cm – max XY offset from original position
YOLO_OBJECT_XY_RANGE_Y = 15.0
YOLO_OBJECT_YAW_RANGE = 360.0     # degrees – uniform [0, range]
YOLO_OBJECT_ROLL_RANGE = 90.0     # degrees – uniform [-range, range]
YOLO_OBJECT_PITCH_MIN = -90.0     # degrees
YOLO_OBJECT_PITCH_MAX = 0.0       # degrees
YOLO_OBJECT_MIN_SEPARATION = 1.0  # cm – minimum gap between objects

# =============================================================================
# YOLO Segmentation  (generate_yolo_seg_data.py)
# =============================================================================

SEG_OUTPUT_FOLDER = "D:/UE5_YOLO_Seg_Data/"
SEG_SEQUENCE_PATH = "/Game/Generated/YOLOSegSequence"
SEG_NUM_SAMPLES = 40
SEG_VAL_SPLIT_RATIO = 0.2
SEG_NEGATIVE_SAMPLE_RATIO = 0.1

SEG_CAM_JITTER_MAX_PITCH = 15.0
SEG_CAM_JITTER_MAX_YAW = 15.0

SEG_MIN_DISTANCE = 100.0      # cm
SEG_MAX_DISTANCE = 400.0      # cm

SEG_RESOLUTION_X = 1280
SEG_RESOLUTION_Y = 720

SEG_WARMUP_FRAMES = 16

SEG_POLYGON_EPSILON_FACTOR = 0.002
SEG_MIN_CONTOUR_AREA = 6
SEG_SAVE_DEBUG_MASKS = False

# =============================================================================
# OBB Detection  (generate_obb_data.py)
# =============================================================================

OBB_OUTPUT_FOLDER = "D:/UE5_OBB_Data/"
OBB_SEQUENCE_PATH = "/Game/Generated/OBBSequence"
OBB_NUM_SAMPLES = 40
OBB_VAL_SPLIT_RATIO = 0.2

OBB_MIN_DISTANCE = 100.0      # cm
OBB_MAX_DISTANCE = 400.0      # cm

OBB_SPATIAL_SAMPLES = 4

# =============================================================================
# DOPE  (generate_dope_data.py)
# =============================================================================

DOPE_OUTPUT_FOLDER = "D:/UE5_DOPE_Data/"
DOPE_SEQUENCE_PATH = "/Game/Generated/DOPESequence"
DOPE_SAMPLES_PER_OBJECT = 500

DOPE_MIN_DISTANCE = 30.0      # cm
DOPE_MAX_DISTANCE = 150.0     # cm

DOPE_SPATIAL_SAMPLES = 4

DOPE_MASK_RESOLUTION_X = 480
DOPE_MASK_RESOLUTION_Y = 270

DOPE_MIN_VISIBILITY_RATIO = 0.20
DOPE_MIN_VISIBLE_PIXELS = 15

DOPE_ENABLE_CAM_JITTER = False
DOPE_CAM_JITTER_MAX_PITCH = 15.0
DOPE_CAM_JITTER_MAX_YAW = 15.0

DOPE_NEGATIVE_SAMPLE_RATIO = 0.10
DOPE_SAVE_DEBUG_MASKS = False

DOPE_RANDOMIZE_OBJECTS = True
DOPE_OBJECT_XY_RANGE_X = 18.0
DOPE_OBJECT_XY_RANGE_Y = 15.0
DOPE_OBJECT_YAW_RANGE = 360.0
DOPE_OBJECT_ROLL_RANGE = 90.0
DOPE_OBJECT_PITCH_MIN = -90.0
DOPE_OBJECT_PITCH_MAX = 0.0
DOPE_OBJECT_MIN_SEPARATION = 1.0   # cm – minimum gap between objects

# =============================================================================
# Test Images  (generate_test_images.py)
# =============================================================================

TEST_OUTPUT_FOLDER = "C:/UE5_Test_Images/"
TEST_SEQUENCE_PATH = "/Game/Generated/TestImageSequence"
TEST_NUM_SAMPLES = 100

TEST_MIN_DISTANCE = 30.0      # cm
TEST_MAX_DISTANCE = 150.0     # cm

TEST_RANDOMIZE_OBJECTS = True
TEST_OBJECT_XY_RANGE_X = 18.0
TEST_OBJECT_XY_RANGE_Y = 15.0
TEST_OBJECT_YAW_RANGE = 360.0
TEST_OBJECT_ROLL_RANGE = 90.0
TEST_OBJECT_PITCH_MIN = -90.0
TEST_OBJECT_PITCH_MAX = 0.0
TEST_OBJECT_MIN_SEPARATION = 1.0   # cm – minimum gap between objects
