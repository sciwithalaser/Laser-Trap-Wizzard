#config.py

#Progress Information
progress_info = {
    "analysis_button_pressed":  True,   # Did the analysis button get pressed?
    "analyzing":                False,  # Is the analysis currently running (turns off if analysis button is pressed)
    "videos_count":             0,      # How many videos are being analyzed?
    "total_steps":              0,      # Total analysis steps with all of the videos counted
    "analyzed_videos":          0,      # Number of videos that were analysed
    "completed_steps":          0,      # Number of steps that were completed
    "current_steps":            0,      # Number of compelted analysis steps in the video currently being analyzed
    "current_total_steps":      0,      # Total number of steps in the analysis of the current video
    "current_progressions":     0,      # Progression of the analysis of the video currently being analysed
    "analyzis_progression":     0,      # Progresison of the total analysis
    "current_task":             "",     # Description of the task currently being performed
    "progress_description":     ""      # Description of the total progress in the analysis
    }

# Default Parameters:
params = {
    "extend" :              10,
    "ccorr_threshold" :     0.4,
    "dist_threshold" :      2,
    "manual_ROI_dim" :      19,
    "annotated_videos" :    True,
    "saveTemplates" :       True,
    "saveCCORR" :           False,
    "saveAnalysisFrames" :  False,
    "savePlots" :           True,
    "erode" :               True
}

# GUI update parameters
gui_update = {
    "thinking_icon" :       ["|", "/", "-", "\\"],
    "current_icon" :        0,
    "update_delay" :        100, # Time in milliseconds before the GUI is updated again.
}

video_anal = {
    
    "video_directories" :       (),     # Tuple with video directories as strings
    "video_loaded" :            {},     # Dictionary remembering if videos have been previously loaded or not
    "rois" :                    {},     # Dictionary. Keys = VideoDir, Values = list of tuples with ROI coordinates and dimanesions
    "templates" :               {},     # Dictionary. Keys = VideoDir, Values = np array of templates for that VideoDir
    "matching_areas" :          {},     # Dictionary. Keys = VideoDir, Values = list of tuples with matching area coordinate and dimensions
    "rightMostROIs" :           {},     # Dictionary. Keys = VideoDir, Value = list of ints with index of the rightMost ROI for ROIs corresponding to each VideoDir.
    "selected_ROI" :            None,
    "current_video_index" :     0,      # Index for navigation across videos.
    "frame" :                   None,   # Initialize empty first frame
    "display_frame" :           None,   # Initialize empty displayed frame (frame that will contain drawn ROIs)
    "template_display_size" :   200,    # Square size of the template display image.
    "total_frames" :            0
}
        