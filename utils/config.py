"""
Configuration parameters for the Wall Extraction Demo
"""

# Model parameters
PATCH_SIZE = 640
DEFAULT_OVERLAP_RATIO = 0.5

# Image processing parameters
SUPPORTED_IMAGE_FORMATS = ['png', 'jpg', 'jpeg']
DEFAULT_DPI = 150  # For PDF conversion

# Visualization parameters
COLOR_MAP = {
    1: (229, 204, 255),  # pink / rooms
    2: (255, 255, 0),    # yellow / walls
    3: (0, 255, 0),      # green / door
    4: (0, 0, 255)       # blue / windows
}

# File paths
CHECKPOINT_PATH = "checkpoints"
RESULTS_PATH = "results"
TEMP_PATH = "temp"

# Streamlit UI parameters
PAGE_TITLE = "Floor Plan Wall Extraction"
PAGE_ICON = "logo/atlas-logo.png"
LOGO_PATH = "logo/atlas-logo-black.png" 