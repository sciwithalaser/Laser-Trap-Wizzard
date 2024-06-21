import customtkinter
from tkinter import filedialog, messagebox
from PIL import Image
import bead_analysis_functions as ba
import cv2
import numpy as np
import webbrowser
import threading
import copy
import os

def fineAdjust(caller, com = "None", stepSize = 1, dimStepSize = 1):

        if caller.video_loaded:

            if caller.selected_ROI is not None:

                video_directory = caller.video_directories[caller.current_video_index]

                # Extract the X- coordinate of the selected ROI
                roi_row, roi_column, w, h, =  caller.selected_ROI

                # Adjust coordiante according to step size and received command
                if com == "down":
                    new_row = roi_row - stepSize
                    updated_ROI = (new_row, roi_column, w, h)
                elif com == "up":
                    new_row = roi_row + stepSize
                    updated_ROI = (new_row, roi_column, w, h)
                elif com == "left":
                    new_column = roi_column - stepSize
                    updated_ROI = (roi_row, new_column, w, h)
                elif com == "right":
                    new_column = roi_column + stepSize
                    updated_ROI = (roi_row, new_column, w, h)
                elif com == "increase":
                    new_dimension = w + dimStepSize
                    updated_ROI = (roi_row, roi_column, new_dimension, new_dimension)
                elif com == "decrease":
                    new_dimension = w - dimStepSize
                    updated_ROI = (roi_row, roi_column, new_dimension, new_dimension)

                # Find selected ROI index in displayed ROIs list.
                index = caller.rois[video_directory].index(caller.selected_ROI)
                caller.rois[video_directory][index] = updated_ROI   
                caller.selected_ROI = updated_ROI

                # Display new frame with ROI
                update_displayed_frame(caller)
            else:
                return

def select_videos(caller, progress_info):       

    # Prompt the user for videos and save their directories in global variable
    caller.video_directories = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4;*.avi")])
    caller.video_loaded.clear()
    caller.selected_ROI = None
    
    if caller.video_directories:
        
        caller.rois = {video_dir: [] for video_dir in caller.video_directories}
        
        # Load the first video in the video_directories list
        caller.current_video_index = 0
        load_video(caller, progress_info)
    
def load_video(caller, progress_info):

    # Extract directory of video that is being loaded
    video_directory = caller.video_directories[caller.current_video_index]

    # Verify if the video was loaded before
    if video_directory not in caller.video_loaded:

        # If the video was not laoded before, update video_loaded status for the video being loaded.
        caller.video_loaded[video_directory] = True
            
    # Opens video being loaded and read the first frame
    cap = cv2.VideoCapture(video_directory)
    ret, caller.frame = cap.read()

    if ret:
        # Display frame with highlighted ROIs
        update_displayed_frame(caller)

        # Update navigtion buttons according to current video index
        check_buttons_state(caller)
    else:
        
        # Update progress description so the user can know that a video was not read
        file_name_without_extension = os.path.splitext(os.path.basename(video_directory))[0]
        progress_info["progress_description"] = f"Error, Unable to read the first frame of the video {file_name_without_extension}."
        caller.master.progress_description.configure(text=progress_info["progress_description"])
        feedback = f"Unable to read the first frame when preparing templates"

        # Remove the video that could not be read element
        temp_list = list(caller.video_directories) # Transform tuple temporarily into a list to do thre removal (can't do it directly on a tuple)
        temp_list.pop(caller.current_video_index)
        caller.video_directories = tuple(temp_list)

        # Remove ROI holder from the video that could not be read
        _ = caller.rois.pop(video_directory)

        # Output an error to the current video analysis folder.
        error_log_path = video_directory + f"-ERROR.txt"       
        with open(error_log_path, 'a') as log_file:
            log_file.write(feedback)

        # Read the next video instead
        caller.current_video_index += 1
        load_video(caller, progress_info)

def update_displayed_frame(caller):

    # Extract directory of video currently loaded
    video_directory = caller.video_directories[caller.current_video_index]

    # Copy original frame to display ROIs
    frame = caller.frame.copy()

    if len(caller.rois[video_directory]) == 2:
        
        # Extract templates based on selected ROIs
        templates, matching_areas, rightMost_ROI = ba.make_templates(frame, caller.rois[video_directory], caller.master.extend)
        caller.templates[video_directory] = templates
        caller.matching_areas[video_directory] = matching_areas
        caller.rightMostROIs[video_directory] = rightMost_ROI

        # Process templates for display        
        for i, template in enumerate(templates):

            pil_template = Image.fromarray(template)
            tk_template = customtkinter.CTkImage(pil_template, size = (caller.template_display_size, caller.template_display_size))
            
            if i == rightMost_ROI:
                caller.template_display_right.configure(image = tk_template, text="", fg_color = "transparent")
            else:
                caller.template_display_left.configure(image = tk_template, text="", fg_color = "transparent")
    else:
        
        # Make one of the template displays black with message
        size = caller.template_display_size
        black_array = np.zeros((size, size))
        pil_black = Image.fromarray(black_array)
        tk_black = customtkinter.CTkImage(pil_black, size = (caller.template_display_size, caller.template_display_size))
        
        # If a single ROI is selected, display the template in one of the template displays
        if len(caller.rois[video_directory]) == 1:

            templates, matching_areas, rightMost_ROI = ba.make_templates(frame, caller.rois[video_directory], caller.master.extend)
            caller.templates[video_directory] = templates
            caller.matching_areas[video_directory] = matching_areas
            caller.rightMostROIs[video_directory] = rightMost_ROI

            pil_template = Image.fromarray(templates[0])
            tk_template = customtkinter.CTkImage(pil_template, size = (caller.template_display_size, caller.template_display_size))
            caller.template_display_left.configure(image = tk_template, text="", fg_color = "transparent")
            caller.template_display_right.configure(image = tk_black, text="Right-most ROI shows up here", fg_color = "black")

        # If no ROIs are selected, display message asking for at least one ROI
        elif len(caller.rois[video_directory]) < 1:

            caller.template_display_left.configure(image = tk_black, text = "Please select at least 1 ROI")
            caller.template_display_right.configure(image = tk_black, text = "Please select at least 1 ROI")
        
        # If more than two ROIs are selected, display a message asking to select at most 2 ROIs
        else:
            caller.template_display_left.configure(image = tk_black, text = "Please select at most 2 ROIs")
            caller.template_display_right.configure(image = tk_black, text = "Please select at most 2 ROIs")

    # Draw Rectangles on first frame to indicate the ROIs
    for roi in caller.rois[video_directory]:

        # Unpack ROI coordinates and shape
        row, column, w, h = roi
        if roi == caller.selected_ROI:
            cv2.rectangle(frame, (column, row), (column + w, row + h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (column, row), (column + w, row + h), (0, 255, 0), 2)
    
    # Process frame to be displayed in the GUI
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, _ = frame.shape
    frame_dimensions = (frame_w, frame_h)
    pil_image = Image.fromarray(frame_rgb)
    tk_image = customtkinter.CTkImage(pil_image, size=frame_dimensions)
    caller.video_display.configure(image=tk_image, text="", fg_color="transparent", width = frame_w, height = frame_h) 

def mouse_left_click(caller, event, manual_ROI_dimensions = 23):
    
    # The function only runs if a video has been loaded.
    if caller.video_loaded:

        # Extracts video directory of the video currently loaded
        video_directory = caller.video_directories[caller.current_video_index]

        insideROIs = []
        for roi in caller.rois[video_directory]:

            # Unpack ROI coordinates and shape
            row, column, w, h = roi

            # Check each ROI for if the user clicked inside of it
            if column < event.x < column + w and row < event.y < row + h:
                insideROIs.append(1)
                caller.selected_ROI = roi
            else:
                insideROIs.append(0)
        
        # If click was outside of any ROI, create new ROI centered around the click
        if 1 not in insideROIs:

            caller.selected_ROI = None

            # Calculate ROI coordinates (uper-left corner of the ROI)
            # Notice that the first dimensions of imaging processing (x, y) is opposite to the dimension of array operations (y,x) 
            roi_row = max(0, event.y - manual_ROI_dimensions // 2) # ROW
            roi_column = max(0, event.x - manual_ROI_dimensions // 2) # COLUMN

            # Add new ROI to displayed ROI list.
            caller.rois[video_directory].append((roi_row, roi_column, manual_ROI_dimensions, manual_ROI_dimensions))

        # Display new frame with ROI
        update_displayed_frame(caller)

def mouse_right_click(caller, event):

    # The function only runs if a video has been loaded.
    if caller.video_loaded:

        # Extracts video directory of the video currently loaded
        video_directory = caller.video_directories[caller.current_video_index]

        # Verify if left click was inside of an ROI
        for roi in caller.rois[video_directory]:

            # Unpack ROI coordinates and shape
            roi_row, roi_column, w, h = roi

            # Check each ROI for if the user clicked inside of it
            if roi_column < event.x < roi_column + w and roi_row < event.y < roi_row + h:
                
                # If the mouse click happened inside ROI, delete the ROI
                caller.rois[video_directory].remove(roi)
                
                break # Break loop once clicked ROI is found
    
        # Display new frame with ROI
        update_displayed_frame(caller)

def check_buttons_state(caller):

    # Configure Previous Video Button
    if caller.current_video_index == 0:
        caller.previous_button.configure(state="disabled")
    else:
        caller.previous_button.configure(state="normal")

    # Configure Next Video Button
    if caller.current_video_index == len(caller.video_directories) - 1:
        caller.next_button.configure(state="disabled")
    else:
        caller.next_button.configure(state="normal")

    # MATCHING EXTEND REGION NAVIGATION
    if caller.master.extend > 0 and caller.master.extend < 20:
        caller.master.settings.matching_region_extend_decrease.configure(state="normal")
        caller.master.settings.matching_region_extend_increase.configure(state="normal")
    if caller.master.extend == 0:
        caller.master.settings.matching_region_extend_decrease.configure(state="disabled")
        caller.master.settings.matching_region_extend_increase.configure(state="normal")
    if caller.master.extend == 20:
        caller.master.settings.matching_region_extend_decrease.configure(state="normal")
        caller.master.settings.matching_region_extend_increase.configure(state="disabled")
    
    # CCORR THRESHOLD SETTING
    if caller.master.ccorr_threshold > 0 and caller.master.ccorr_threshold < 1:
        caller.master.settings.ccorr_threshold_decrease.configure(state="normal")
        caller.master.settings.ccorr_threshold_increase.configure(state="normal")
    if caller.master.ccorr_threshold == 0:
        caller.master.settings.ccorr_threshold_decrease.configure(state="disabled")
        caller.master.settings.ccorr_threshold_increase.configure(state="normal")
    if caller.master.ccorr_threshold == 1:
        caller.master.settings.ccorr_threshold_decrease.configure(state="normal")
        caller.master.settings.ccorr_threshold_increase.configure(state="disabled")
    
    # DIST THRESHOLD
    if caller.master.dist_threshold > 1 and caller.master.dist_threshold < caller.master.extend:
        caller.master.settings.dist_threshold_decrease.configure(state="normal")
        caller.master.settings.dist_threshold_increase.configure(state="normal")
    if caller.master.dist_threshold == 1:
        caller.master.settings.dist_threshold_decrease.configure(state="disabled")
        caller.master.settings.dist_threshold_increase.configure(state="normal")
    if caller.master.dist_threshold == caller.master.extend:
        caller.master.settings.dist_threshold_decrease.configure(state="normal")
        caller.master.settings.dist_threshold_increase.configure(state="disabled")
    
    # ROI DIMENSION SETTING
    if caller.master.manual_ROI_dim > 1:
        caller.master.settings.manual_ROI_dim_decrease.configure(state="normal")
        caller.master.settings.manual_ROI_dim_increase.configure(state="normal")
    if caller.master.manual_ROI_dim == 1:
        caller.master.settings.manual_ROI_dim_decrease.configure(state="disabled")
        
def next_video(caller, progress_info):

    # Increases current_video_index by one if not the last video
    if caller.current_video_index < len(caller.video_directories) - 1:
        caller.current_video_index += 1

        caller.selected_ROI = None

        # Load video with the new current_video_index
        load_video(caller, progress_info)

def previous_video(caller, progress_info):

    # Increases current_video_index by one if not the last video
    if caller.current_video_index > 0:
        caller.current_video_index -= 1

        caller.selected_ROI = None

        # Load video with the new current_video_index
        load_video(caller, progress_info)

def increaseButton_callback(master, parameter_name, display):
    
    if parameter_name == "ccorr_threshold":
        setattr(master, parameter_name, getattr(master, parameter_name) + 0.1)
        display.configure(text=f"{getattr(master, parameter_name):.1f}")

    else:
        setattr(master, parameter_name, getattr(master, parameter_name) + 1)
        display.configure(text=f"{getattr(master, parameter_name)}")

    check_buttons_state(master.video_analysis_frame)

def decreaseButton_callback(master, parameter_name, display):

    if parameter_name == "ccorr_threshold":
        setattr(master, parameter_name, getattr(master, parameter_name) - 0.1)
        display.configure(text=f"{getattr(master, parameter_name):.1f}")
    else:    
        setattr(master, parameter_name, getattr(master, parameter_name) - 1)
        display.configure(text=f"{getattr(master, parameter_name)}")

    check_buttons_state(master.video_analysis_frame)

def watermark_callback(url):

    webbrowser.open_new(url)

def template_selection(caller, event, templateLabel):
    
    # The function only runs if a video has been loaded.
    if caller.video_loaded:

        # Extracts video directory of the video currently loaded
        video_directory = caller.video_directories[caller.current_video_index]

        max_column_coordinate = 0
        for i,roi in enumerate(caller.rois[video_directory]):
            
            # Unpack ROI coordinates
            _,roi_column,_,_ = roi

            # Determine right most ROI
            if max_column_coordinate < roi_column:
                max_column_coordinate = roi_column
                rightMost_ROI = i
            
        for roi in caller.rois[video_directory]:
            if roi == caller.rois[video_directory][rightMost_ROI] and templateLabel == "right":
                caller.selected_ROI = roi
            elif roi != caller.rois[video_directory][rightMost_ROI] and templateLabel == "left":
                caller.selected_ROI = roi

        # Display new frame with ROI
        update_displayed_frame(caller)

def analyzis_button_callback(caller, progress_info):

    # Check if analysis is running
    if progress_info["Analyzing"] == False:

        # First, verify if any videos were loaded.
        if not caller.video_analysis_frame.video_directories:
            messagebox.showwarning(title = "WARNING", message="Please load at least one video for analysis")
            return

        # Create list of errors to be solved before analysis.
        errors = {}
        any_error = False

        # Verify if correct number of ROIs is selected for each loaded video
        for index, rois in enumerate(caller.video_analysis_frame.rois.values()):
            
            current_video_number = index + 1
            # Verify if at least one ROI is selected
            if len(rois) < 1:
                errors[current_video_number] = f"Ensure that at least 1 ROI is selected in video number {current_video_number}"
            if len(rois) > 2:
                errors[current_video_number] = f"Ensure that at most 2 ROIs are selected in video number {current_video_number}"
        
        for error in errors.values():
            if error:
                any_error = True
                break
        
        # Verify if any error were found
        if any_error:
            error_prompt = "Please, solve the following errors before analysis:\n\n"
            for error in errors.values():
                if error:
                    error_prompt = error_prompt+" - "+error+"\n"
            messagebox.showwarning(title = "WARNING", message=error_prompt)
        
        # If no errors were found, analyse
        else:

            # Update Analyzing Progress_Info
            progress_info["analysis_button_pressed"] = True
            progress_info["Analyzing"] = True          
            caller.update_gui()

            # copy immutable and deepCopy mutable data to pass into new thread for video analysis
            video_directories = caller.video_analysis_frame.video_directories # List
            templates = copy.deepcopy(caller.video_analysis_frame.templates) # Dictionary
            matching_areas = copy.deepcopy(caller.video_analysis_frame.matching_areas) # Dictionary
            roi_coordinates = copy.deepcopy(caller.video_analysis_frame.rois) # Dictionary
            rightMostROIs = copy.deepcopy(caller.video_analysis_frame.rightMostROIs) # Dictionary
            annotatedVideos = caller.annotated_videos
            saveTemplates = caller.saveTemplates
            saveCCORR = caller.saveCCORR
            saveAnalysisFrames = caller.saveAnalysisFrames
            savePlots =  caller.savePlots
            ccorr_thresh = caller.ccorr_threshold
            dist_thresh = caller.dist_threshold

            # Start analysis on a separate thread so that GUI can continue being updated
            analysis_thread = threading.Thread(target=analyze, args=(video_directories, templates, matching_areas, roi_coordinates, rightMostROIs, annotatedVideos, saveTemplates, saveCCORR, saveAnalysisFrames, savePlots, ccorr_thresh, dist_thresh, progress_info))
            analysis_thread.start()
    else:
        progress_info["Analyzing"] = False
        progress_info["progress_description"] = "Analysis Cancelled"

def analyze(video_directories, templates, mathcing_areas, roi_coordinates, rightMostROIs, annotatedVideos, saveTemplates, saveCCORR, saveAnalysisFrames, savePlots, ccorr_thresh, dist_thresh, progress_info):

    # Analyze one video at a time
    for video_directory in video_directories:
        
        # Count total number of frames for progress tracking
        total_steps = 0
        for directory in video_directories:

            cap = cv2.VideoCapture(directory)
            if not cap.isOpened():
                error_prompt = "Could not read videos when setting up variables for progress track"
                messagebox.showwarning(title = "WARNING", message=error_prompt)
                continue
            
            template_counts = len(templates[directory])
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_steps += frame_count * template_counts
            if annotatedVideos == True:
                total_steps += frame_count
            cap.release()
        
        progress_info["total_steps"] = total_steps
        progress_info["videos_count"] = len(video_directories)

        # Verify if cancel button was not pressed before continuing
        if progress_info["Analyzing"] == True:
            
            # Update progress info
            progress_info["analyzed_videos"] += 1
            progress_info["current_steps"] = 0
            progress_info["current_total_steps"] = 0

            currentTemplates = templates[video_directory]
            currentMatchingAreas = mathcing_areas[video_directory]
            current_ROI_coordinates = roi_coordinates[video_directory]
            current_rightMostROI = rightMostROIs[video_directory]

            ba.analyze_video(progress_info, video_directory, currentTemplates, currentMatchingAreas, current_ROI_coordinates, current_rightMostROI, annotatedVideos, saveTemplates, saveCCORR, saveAnalysisFrames, savePlots, ccorr_thresh, dist_thresh)
        
        else:
            progress_info["progress_description"] = "Analysis CANCELLED"
            return
    
    progress_info["progress_description"] = "Analysis COMPLETED"
    progress_info["Analyzing"] = False

def annotatedVideos_callback(caller):

    if caller.annotated_videos.get() ==1 :
        caller.master.annotated_videos = True
    else:
        caller.master.annotated_videos = False

def saveTemplates_callback(caller):

    if caller.saveTemplates.get() == 1:
        caller.master.saveTemplates = True
    else:
        caller.master.saveTemplates = False

    print(f"The save template option is set to {caller.master.saveTemplates}")

def saveCCORR_callback(caller):

    if caller.saveTemplates.get() == 1:
        caller.master.saveCCORR = True
    else:
        caller.master.saveCCORR = False

def saveAnalysisFrames_callback(caller):

    if caller.saveTemplates.get() == 1:
        caller.master.saveAnalysisFrames = True
    else:
        caller.master.saveAnalysisFrames = False

def reset_settings(caller):

    # Reset all parameters to default values
    caller.master.extend = 10
    caller.master.ccorr_threshold = 0.4
    caller.master.dist_threshold = 2
    caller.master.manual_ROI_dim = 23
    caller.master.autoTemplate = True
    caller.master.annotated_videos = True
    caller.master.saveTemplates = False

    # Reset all diplays
    caller.matching_region_extend_display.configure(text=f"{caller.master.extend}")
    caller.cccorr_threshold_display.configure(text=f"{caller.master.ccorr_threshold}")
    caller.dist_threshold_display.configure(text=f"{caller.master.dist_threshold}")
    caller.manual_ROI_dim_display.configure(text=f"{caller.master.manual_ROI_dim}")


    # Reset all switches
    caller.annotated_videos.select()
    caller.saveTemplates.deselect()
    caller.saveAnalysisFrames.deselect()
    caller.saveCCORR.deselect()
    caller.savePlots.select()