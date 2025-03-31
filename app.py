import os
from PIL import Image
import streamlit as st
from ultralytics import YOLO
from utils.utils import (create_patches, 
                  generate_patches_masks, merging_patches, visualize_mask)
from utils.config import *  # Import configurations
import cv2

# Get the absolute path to the WallExtractionDemo directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Page configuration
st.set_page_config(
    layout="wide",
    page_title=PAGE_TITLE,
    page_icon=os.path.join(BASE_DIR, PAGE_ICON)
)

# Create a container for logo and title
title_col1, title_col2 = st.columns([0.1, 0.9])  # Adjust ratio as needed for logo size

with title_col1:
    st.image(os.path.join(BASE_DIR, LOGO_PATH), width=100)  # Adjust width as needed

with title_col2:
    st.title(PAGE_TITLE)

st.divider()
########################################################
st.header("Upload Blueprint Image")

upload_col, display_col = st.columns([1, 2]) 

with upload_col:
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    #uploaded_file = "Cleaned_3BF9CE05279C4823B9A7D21A02023C51.jpg"

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            
        # Display width and height
        width, height = image.size
        
        # Display file and image information
        st.subheader("File Information:")
        st.write(f"• Filename: {uploaded_file.name}")
        st.write(f"• Dimensions: {width} x {height} pixels")
        st.write(f"• Aspect ratio: {width/height:.2f}")
        st.write(f"• Image format: {image.format}")
        st.write(f"• Color mode: {image.mode}")
        
        # Calculate and display file size
        if isinstance(uploaded_file, str):  # If using hardcoded path
            file_size = os.path.getsize(uploaded_file) / (1024 * 1024)  # Convert to MB
        else:  # If using file uploader
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Convert to MB
        st.write(f"• File size: {file_size:.2f} MB")

with display_col:
    if uploaded_file is not None:
        # Display uploaded image
        st.image(image, caption="Uploaded Blueprint", use_container_width=True)
        
st.divider()
########################################################
st.header("Configurations")

col1, col2, col3 = st.columns(3)
with col1:
    # Checkpoint selection dropdown
    CHECKPOINT_PATH = "checkpoints"
    checkpoint_list = [f for f in os.listdir(CHECKPOINT_PATH) 
                      if os.path.isdir(os.path.join(CHECKPOINT_PATH, f)) 
                      and f != '.DS_Store']    
    checkpoint_list.insert(0, "")
    checkpoint = st.selectbox("Select model checkpoint:", checkpoint_list)

    
with col2:
    new_img_size = st.selectbox("Resize:", ["No resize", "1024", "2048"])

with col3:
    # Overlapping ratio slider
    overlap_ratio = st.slider("Overlapping ratio:", 
                          min_value=0.0, 
                          max_value=1.0, 
                          value=0.5, 
                          step=0.1)
    
button_wall_extraction = st.button("Start Wall Extraction", type="primary")


st.divider()
########################################################

if button_wall_extraction:
    st.header("Wall Extraction Results")
    st.write("Wall Extraction Started")

    # Load model checkpoint
    model = YOLO(os.path.join(CHECKPOINT_PATH, checkpoint, "weights/best.pt"))

    # Handle image resizing
    if new_img_size == "No resize":
        processed_image = image
        processed_width, processed_height = width, height
    else:
        new_size = int(new_img_size)
        processed_image = image.resize((new_size, new_size))
        processed_width = processed_height = new_size

    # Create patches
    patches, step = create_patches(processed_image, patch_size=640, overlap_ratio=overlap_ratio)

    # Add progress bar
    progress_bar = st.progress(0, text="Starting Walls Extraction...")
    
    # Process patches with progress bar
    patches_masks_dic = generate_patches_masks(patches, model, progress_bar)

    # Get mask for processed image
    processed_mask = merging_patches(processed_width, processed_height, patches_masks_dic, step)
    
    # Project back to original dimensions only if resizing was applied
    if new_img_size != "No resize":
        final_mask = cv2.resize(processed_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        final_mask = processed_mask

    # Get filename without extension for result name
    base_filename = os.path.splitext(uploaded_file.name)[0]
    result_name = f"{base_filename}_PatchRatio-{overlap_ratio}_Checkpoints-{checkpoint}"
    
    os.makedirs('results', exist_ok=True)
    
    path_to_results = os.path.join('results', result_name + '.png')
    mask_overlay = visualize_mask(final_mask, image, path_to_results)
    
    # Display color labels
    st.markdown("""
    <style>
    .color-label {
        display: inline-block;
        width: 15px;
        height: 15px;
        margin-right: 8px;
        vertical-align: middle;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Color Legend:**
    - <div class="color-label" style="background-color: rgb(229, 204, 255); display: inline-block"></div> Rooms
    - <div class="color-label" style="background-color: rgb(255, 255, 0); display: inline-block"></div> Walls
    - <div class="color-label" style="background-color: rgb(0, 255, 0); display: inline-block"></div> Doors
    - <div class="color-label" style="background-color: rgb(0, 0, 255); display: inline-block"></div> Windows
    """, unsafe_allow_html=True)
    
    st.image(mask_overlay, caption="Segmentation Overlay", use_container_width=True)



