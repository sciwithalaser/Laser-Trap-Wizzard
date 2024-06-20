import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import griddata
import os

def make_templates(frame, roi_coordinates, extend):

    """
    Generate templates and matching regions for template matching based on regions of interest (ROIs) in the given frame.

    Parameters:
    ----------
    frame : np.ndarray
        The image frame from which templates will be extracted.
    roi_coordinates : list of tuples
        A list of tuples where each tuple contains four integers (row, column, width, height)
        representing the coordinates and size of each region of interest (ROI) in the frame.
    extend : int
        The number of pixels to extend the matching region beyond the boundaries of the ROI.

    Returns:
    -------
    templates : list of np.ndarray
        A list of templates extracted from the frame, one for each ROI. Each template is a sub-array
        of the processed frame corresponding to the coordinates and size of the ROI.
    matching_regions : list of tuples
        A list of tuples where each tuple contains four integers (row, column, width, height)
        representing the extended matching region around each ROI.
    rightMost_ROI : int
        The index of the ROI that is located furthest to the right in the frame, based on the column coordinate.

    Notes:
    -----
    - The function processes the input frame using a predefined `process_image` function before extracting templates.
    - The `extend` parameter is used to increase the size of the matching region around each ROI.
    """

    # Initialize lists to contain templates and mathing regions
    templates = [[] for _ in roi_coordinates]
    matching_regions = [[] for _ in roi_coordinates]

    max_column_coordinate = 0 # Variable required to find the rightMost_ROI
    
    for i, roi in enumerate(roi_coordinates):
        
        # Unpack ROI coordinates
        roi_row, roi_column, roi_width, roi_height = roi

        # Update right most ROI
        if max_column_coordinate < roi_column:
            max_column_coordinate = roi_column
            rightMost_ROI = i

        # Define matching region
        matchingRegion_row = roi_row - extend # Row
        matchingRegion_column = roi_column - extend # Column
        matchingRegion_width = roi_width + extend * 2
        matchingRegion_height = roi_height + extend * 2
        matching_regions[i] = (matchingRegion_row, matchingRegion_column, matchingRegion_width, matchingRegion_height)
        
        # Process frame for template extraction
        processed_frame = process_image(frame)
        
        # Template extraction:
        templates[i] = processed_frame[roi_row : roi_row + roi_height, roi_column : roi_column + roi_width]
        
    return templates, matching_regions, rightMost_ROI

def analyze_video(progress_info, video_directory, templates, matching_areas, ROI_coordinates, rightMost_ROI, annotatedVideos = True, saveTemplates = True, saveCCORR = True, saveAnalysisFrames = True, savePlots = True, ccorr_thresh = 0.4, dist_thresh = 2):
    
    """
    Analyzes a video to track bead displacements using template matching and generates various outputs including CSV files, plots, and annotated videos.

    Parameters:
    ----------
    progress_info : dict
        A dictionary to keep track of the analysis progress and status.
    video_directory : str
        The path to the video file to be analyzed.
    templates : list of np.ndarray
        A list of templates to be used for template matching.
    matching_areas : list of tuples
        A list of tuples, each containing the coordinates and dimensions of the matching areas for each template.
    ROI_coordinates : list of tuples
        A list of tuples, each containing the coordinates and dimensions of the regions of interest (ROIs).
    rightMost_ROI : int
        The index of the ROI that is located furthest to the right in the frame.
    annotatedVideos : bool, optional
        Whether to generate an annotated video showing the analysis results (default is True).
    saveTemplates : bool, optional
        Whether to save the templates as image files (default is True).
    saveCCORR : bool, optional
        Whether to save the cross-correlation matrices as a video (default is True).
    saveAnalysisFrames : bool, optional
        Whether to save the analysis frames as a video (default is True).
    savePlots : bool, optional
        Whether to save displacement plots as image files (default is True).
    ccorr_thresh : float, optional
        The correlation threshold for template matching (default is 0.4).
    dist_thresh : float, optional
        The distance threshold for determining bead location changes (default is 2).

    Returns:
    -------
    None
        The function performs analysis and generates output files but does not return any value.

    Notes:
    -----
    - The function processes the video frame-by-frame, performing template matching to track bead displacements.
    - Generates CSV files for bead locations and displacements.
    - Optionally generates videos for cross-correlation matrices, analysis frames, and an annotated video showing the analysis results.
    - Optionally saves displacement plots as image files.
    - The progress of the analysis is tracked and updated in the `progress_info` dictionary.
    """
        
    # Analysis Parameters
    CCORR_THRESHOLD = ccorr_thresh              # Minimum CCORR correlation threshold 
    DIST_THRESHOLD = dist_thresh*np.sqrt(2)     # Maximum distance between new bead location and previous bead location (two diagonal pixels)
    CONVERSION_FACTOR = 5.5                     # Pixels per micrometer

    # Initializing lists to contain data
    beadLocations = [[] for _ in templates] # Pixel location of beads in each frame (one list per template)
    fineBeadLocations = [[] for _ in templates] # Pixel location of beads in each frame in ressampled grid (one list per template)
    dataFrameNumbers = [[] for _ in templates] # Frames contaning the bead location.
 
    # Creating Output Path
    file_name_without_extension = os.path.splitext(os.path.basename(video_directory))[0]
    slideID = file_name_without_extension.split("-", 1)[0]
    output_path = os.path.dirname(video_directory) + "/" + slideID + "-ANALYSIS"
    if not os.path.exists(output_path):
        os.makedirs(output_path)    

    # Analyze with one template at a time
    for i, (template, matching_area) in enumerate(zip(templates, matching_areas)):      

        # Open the video but don't read it
        cap = cv2.VideoCapture(video_directory)
        if not cap.isOpened():
            Error = f"Error: Unable to open video {video_directory}"
            progress_info["progress_description"] = Error
            return
        
        # Get Frame Rate for plotting later
        frameRate = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        progress_info["current_total_steps"] = frame_count*len(templates)
        progress_info["current_task"] = f"Template Matching Template {i+1} of {len(templates)}"
        
        # Read first frame
        ret, frame_now = cap.read()
        if not ret:    
            Error = f"Error: Unable to read first frame from video {slideID} when matching template {i+1}."
            progress_info["progress_description"] = Error
            return
        
        if saveCCORR == True:
            ccorr_video_path = output_path + "/" + file_name_without_extension + f"Bead_{i+1}_CCOR_VID.mp4" 
            frame_height, frame_width, _ = frame_now.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            ccorr_out = cv2.VideoWriter(ccorr_video_path, fourcc, frameRate, (frame_width, frame_height))
        else:
            ccorr_out = None
        
        if saveAnalysisFrames == True:
            analysisFrames_video_path = output_path + "/" + file_name_without_extension + f"Bead_{i+1}_ANAL_VID.mp4" 
            frame_height, frame_width, _ = frame_now.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            analFrame_out = cv2.VideoWriter(analysisFrames_video_path, fourcc, frameRate, (frame_width, frame_height))
        else:
            analFrame_out = None

        # Define origin coordinate
        roi = ROI_coordinates[i]
        originLoc = (roi[0], roi[1]) # (Row, Column)
        
        # Set original match location to origin location
        matchLoc = originLoc

        # Define the number of the initial frame.
        currentFrameNumber = 1

        # Loop through every Frame
        while cap.isOpened():
       
            # Verify that analysis was not cancelled before proceeding
            if progress_info["Analyzing"] == False:
                progress_info["progress_description"] = "Analysis Cancelled"
                return

            # Template matching on current Frame with current template
            matchLoc, fineLoc, timeLoc, currentFrameNumber = template_matching(progress_info,
                frame_now, currentFrameNumber, cap, matchLoc, template, matching_area, CCORR_THRESHOLD, DIST_THRESHOLD, ccorr_out=ccorr_out, analFrame_out=analFrame_out)
            
            # Append data to their respective data lists.
            beadLocations[i].append(matchLoc)
            fineBeadLocations[i].append(fineLoc)
            dataFrameNumbers[i].append(timeLoc)

            # Increment currentFrameNumber each time a new frame is read.
            currentFrameNumber += 1

            # Read the next frame
            ret, frame_now = cap.read()
            if not ret:
                if currentFrameNumber < frame_count:
                    Error = f"Error: Unable to read the next frame from video {slideID} when matching template {i+1}."
                    progress_info["progress_description"] = Error
                break

            # Complete an analysis step each time a frame is analyzed.
            progress_info["current_steps"] += 1
            progress_info["completed_steps"] += 1

            current_steps = progress_info["current_steps"]
            current_total_steps = progress_info["current_total_steps"]
            completed_steps = progress_info["completed_steps"]
            total_steps = progress_info["total_steps"]
            videos_count = progress_info["videos_count"]

            progress_info["current_progressions"] = current_steps/current_total_steps
            progress_info["analyzis_progression"] = completed_steps/total_steps

            current_progressions = progress_info["current_progressions"]
            analyzis_progression = progress_info["analyzis_progression"]
            analyzed_videos = progress_info["analyzed_videos"]
            current_task = progress_info["current_task"]

            progress_info["progress_description"] = f"Analyzing Video {analyzed_videos} of {videos_count} ({current_progressions * 100:.1f}%) | Current Task: {current_task} | Total Progression: {analyzis_progression*100:.1f}%"

        # Release video reader and video writers.  
        cap.release()
        if ccorr_out is not None:
            ccorr_out.release()
        if analFrame_out is not None:    
            analFrame_out.release()

    # Write CSVs
    progress_info["current_task"] = f"Saving Data into CSVs"

    # Write beadLocations data into CSVs
    for i, locationData in enumerate(beadLocations):
        
        # Combine frame_number list and data into list of tuples.
        location_frame_pairs = list(zip(dataFrameNumbers[i], locationData))

        # Define path of the CSV file
        csv_output_path = output_path + "/" + file_name_without_extension + f"-Bead_{i+1}_LOCS.csv"

        # Write to CSV
        write_to_CSV(location_frame_pairs, csv_output_path)

    # Write fineBeadLocation into CSVs
    for i, locationData in enumerate(fineBeadLocations):
        
        # Combine frame_number list and data into list of tuples.
        location_frame_pairs = list(zip(dataFrameNumbers[i], locationData))

        # Define path of the CSV file
        csv_output_path = output_path + "/" + file_name_without_extension + f"-Bead_{i+1}_FINELOCS.csv"
        
        # Write to CSV
        write_to_CSV(location_frame_pairs, csv_output_path)
        
    # Convert locations data into displacement data.
    fineDisplacementData = calculateDistances(fineBeadLocations)

    # Convert pixel displacement data into microns displacement data
    fineMicronDisplacementData = [[pixel_value/CONVERSION_FACTOR for pixel_value in bead] for bead in fineDisplacementData]

    # Write displacement data for each bead into separate CSVs and plot
    for i,displacement_data in enumerate(fineMicronDisplacementData):
        
        # Combine frame_number list and data into list of tuples.
        displacement_frame_pairs = list(zip(dataFrameNumbers[i], displacement_data))

        # Define path of the CSV file
        csv_output_path = output_path + "/" + file_name_without_extension + f"-Bead_{i+1}_DISPS.csv"

        # Write to CSV
        write_to_CSV(displacement_frame_pairs, csv_output_path)

        if savePlots == True:

            progress_info["current_task"] = f"Plotting Data"

            # Define path of individual plots
            individual_plots_path = output_path + "/" + file_name_without_extension + f"-Bead_{i+1}_PLOT.png"
            plot_data(displacement_frame_pairs, frameRate, individual_plots_path)

            plt.figure("Joined")

            # Plot joined plot
            joined_plot_path = output_path + "/" + file_name_without_extension + f"-JOINED_PLOT.png"

            frames = np.array([item[0] for item in displacement_frame_pairs])
            displacements = np.array([item[1] for item in displacement_frame_pairs])

            # Smooth out the data
            window_size = 5
            displacements = np.convolve(displacements, np.ones(window_size)/window_size, mode='same')

            # Change frame number to seconds.
            seconds = frames / frameRate

            plt.plot(seconds, displacements, label=f"Bead {i+1}")
    
    if savePlots == True:

        # Add labels and title
        plt.xlabel('Time (seconds)')
        plt.ylabel('Displacement (μm)')
        plt.legend()

        # Save the plot as a figure
        plt.savefig(joined_plot_path)
        plt.close('all')         
        
    # Create annotated video if required
    if annotatedVideos == True:
        output_video_path = output_path + "/" + file_name_without_extension + f"-ANNOT_VID.mp4"
        annotate_video(progress_info, video_directory, beadLocations, fineBeadLocations, dataFrameNumbers, ROI_coordinates, rightMost_ROI, output_video_path)

    # Save templates if requested
    if saveTemplates == True:
        print("I just tried saving a template")
        for i,template in enumerate(templates):
            templatePath = output_path + "/" + file_name_without_extension + f"-Template_{i+1}.jpg"
            cv2.imwrite(templatePath, template)

def template_matching(progress_info, frame, analysisFrameNumber, cap, previousMatchLoc, template, matching_area, ccorr_threshold, dist_threshold, ccorr_out = None, analFrame_out = None):

    """
    Performs template matching on a video frame to track bead displacement and writes the correlation matrices and analysis frames to videos if specified.

    Parameters:
    ----------
    progress_info : dict
        A dictionary to keep track of the analysis progress and status.
    frame : np.ndarray
        The current video frame to be analyzed.
    analysisFrameNumber : int
        The number of the current frame being analyzed.
    cap : cv2.VideoCapture
        The video capture object to read frames from the video.
    previousMatchLoc : tuple
        The coordinates of the previously matched location (row, column).
    template : np.ndarray
        The template to be used for matching.
    matching_area : tuple
        A tuple containing the coordinates and dimensions of the matching area (row, column, width, height).
    ccorr_threshold : float
        The correlation threshold for template matching.
    dist_threshold : float
        The distance threshold for determining bead location changes.
    ccorr_out : cv2.VideoWriter, optional
        The video writer object to write the cross-correlation matrices (default is None).
    analFrame_out : cv2.VideoWriter, optional
        The video writer object to write the analysis frames (default is None).

    Returns:
    -------
    bestMatchLoc : tuple
        The coordinates of the best match location in the current frame (row, column).
    fineLoc : tuple
        The fine-tuned coordinates of the best match location in a finer grid.
    timeLoc : int
        The frame number corresponding to the middle of all summed frames.
    currentFrameNumber : int
        The number of the current frame after processing.

    Notes:
    -----
    - The function processes the input frame, performs template matching, and updates the progress information.
    - Writes the cross-correlation matrices and analysis frames to the specified video files if provided.
    - If a good match is not found in the current frame, the function sums subsequent frames until a good match is found or a limit is reached.
    # Pre process the frame before template matching:
    analysis_frame = process_image(frame)
    currentFrameNumber = analysisFrameNumber
    """

    # Create matching_area mask:
    top_left_matching_area = (matching_area[0], matching_area[1])
    bottom_right_matching_area = (matching_area[0] + matching_area[3],  matching_area[1] + matching_area[2])
    mask = np.zeros_like(analysis_frame)
    mask[top_left_matching_area[0]:bottom_right_matching_area[0]+1, top_left_matching_area[1]:bottom_right_matching_area[1]+1] = 1

    # Mask analysis frame
    analysis_frame = analysis_frame * mask

    # Ensure that analysis frame has the same dtype as the template for proper template matching
    analysis_frame = np.array(analysis_frame, dtype=template.dtype)

    if analFrame_out is not None:

        # Convert to RGB for video writer
        out_analFrame = cv2.cvtColor(analysis_frame, cv2.COLOR_GRAY2RGB)

        # Write the analysis frame
        analFrame_out.write(out_analFrame)

    # Perform normalized cross-correlation with respective template
    ccorr_matrix = cv2.matchTemplate(analysis_frame, template, cv2.TM_CCORR_NORMED)
    
    if ccorr_out is not None:
        
        # Normalize the correlation matrix to the range [0, 255] and convert to uint8
        ccorr_normalized = cv2.normalize(ccorr_matrix, None, 0, 255, cv2.NORM_MINMAX)
        ccorr_normalized = np.uint8(ccorr_normalized)
        
        # Resize to match the original frame size if necessary
        if ccorr_normalized.shape != analysis_frame.shape:
            ccorr_normalized = cv2.resize(ccorr_normalized, (analysis_frame.shape[1], analysis_frame.shape[0]))

        # Convert to RGB for video writer
        out_ccorr_matrix = cv2.cvtColor(ccorr_normalized, cv2.COLOR_GRAY2RGB)

        # Write the CCORR matrix
        ccorr_out.write(out_ccorr_matrix)

    # Get sorted list of ccorr_value and location pairs as well as the index in this list of the best match (based on ccorr_threshold and dist_threshold)
    sorted_ccorr_loc_pair, matchIndex = find_best_match(ccorr_matrix, previousMatchLoc, ccorr_threshold, dist_threshold)

    if matchIndex == -1:
        
        # Initialize summator
        summator = analysis_frame.copy()
        summedFrames = 1

        while matchIndex == -1:
            
            # If a whole second was summed and a good match was not found, reset the summator and reassingn analysis frame number.
            if summedFrames > 30:
                summator = next_frame.copy()
                summedFrames = 1
                analysisFrameNumber = currentFrameNumber

            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                Error = f"Error reading frame {currentFrameNumber} when adding it to the summator"
                progress_info["progress_description"] = Error
                return
            
            # Complete an analysis step each time a frame is analyzed.
            progress_info["current_steps"] += 1
            progress_info["completed_steps"] += 1

            current_steps = progress_info["current_steps"]
            current_total_steps = progress_info["current_total_steps"]
            completed_steps = progress_info["completed_steps"]
            total_steps = progress_info["total_steps"]
            videos_count = progress_info["videos_count"]

            progress_info["current_progressions"] = current_steps/current_total_steps
            progress_info["analyzis_progression"] = completed_steps/total_steps

            current_progressions = progress_info["current_progressions"]
            analyzis_progression = progress_info["analyzis_progression"]
            analyzed_videos = progress_info["analyzed_videos"]
            current_task = progress_info["current_task"]

            progress_info["progress_description"] = f"Analyzing Video {analyzed_videos} of {videos_count} ({current_progressions * 100:.1f}%) | Current Task: {current_task} | Total Progression: {analyzis_progression*100:.1f}%"
            
            # Update current frame number
            currentFrameNumber += 1
            
            # Process next frame
            next_frame = process_image(frame)
            next_frame = np.array(next_frame, dtype=template.dtype)
            
            # Sum next frame to summator and update summedFrames.
            summator = summator + next_frame
            summedFrames += 1

            # Mask summator for template matching
            summator = summator * mask

            summator = np.array(summator, dtype=template.dtype)

            # Perform normalized cross-correlation with summator and find the best match
            ccorr_matrix = cv2.matchTemplate(summator, template, cv2.TM_CCORR_NORMED)

            sorted_ccorr_loc_pair, matchIndex = find_best_match(ccorr_matrix, previousMatchLoc, ccorr_threshold, dist_threshold)

    # Extract the location of the best match
    bestMatchLoc = sorted_ccorr_loc_pair[matchIndex][1] # (ROW, COLUMN)

    # Extract fine location in finer grid
    fineLoc, _ = resample_surface(ccorr_matrix, bestMatchLoc)

    # Assigned location data to the middle frame of all summed frames
    timeLoc = int((analysisFrameNumber + currentFrameNumber)/2)
    
    # Retunr the best match location and the fine best-match-loation as coordinates in the frame, return last frame that was read. 
    return bestMatchLoc, fineLoc, timeLoc, currentFrameNumber

def annotate_video(progress_info, video_directory, beadLocations, fineBeadLocations, dataFrameNumbers, ROI_coordinates, rightMost_ROI, output_video_path):
    
    """
    Annotates a video with bead locations and writes the annotated video to the specified output path.

    Parameters:
    ----------
    progress_info : dict
        A dictionary to keep track of the annotation progress and status.
    video_directory : str
        The path to the video file to be annotated.
    beadLocations : list of list of tuples
        A nested list where each sublist contains tuples of (row, column) coordinates for the bead locations in each frame.
    fineBeadLocations : list of list of tuples
        A nested list where each sublist contains tuples of fine-tuned (row, column) coordinates for the bead locations in each frame.
    dataFrameNumbers : list of list of int
        A nested list where each sublist contains frame numbers corresponding to the bead locations.
    ROI_coordinates : list of tuples
        A list of tuples, each containing the coordinates and dimensions of the regions of interest (ROIs).
    rightMost_ROI : int
        The index of the ROI that is located furthest to the right in the frame.
    output_video_path : str
        The path where the annotated video will be saved.

    Returns:
    -------
    None
        The function performs annotation and writes the output video but does not return any value.

    Notes:
    -----
    - The function processes the video frame-by-frame, annotating each frame with bead locations if they are found.
    - Updates the progress information during the annotation process.
    - Saves the annotated video to the specified output path.
    """
    # Update progress info description
    progress_info["current_task"] = f"Writing Annotated Video"
    current_task = progress_info["current_task"]
        
    # Extract origin coordinates
    origin_coordinates = [locationsList[0] for locationsList in beadLocations]

    # Open video to read frame to annotate
    cap = cv2.VideoCapture(video_directory)
    if not cap.isOpened():
        Error = f"Error: Unable to open video {video_directory}"
        progress_info["progress_description"] = Error
        return
    
    # Get frame rate and frame count of annotated video
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        Error = f"Error: Unable to read frames while annotating {video_directory}"
        progress_info["progress_description"] = Error
        return
    
    # Get frame width and height from the first frame
    frame_height, frame_width, _ = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))
    
    current_frame = 0
    while cap.isOpened():
        
        # Keep track of the frame read
        current_frame += 1

        # Copy original frame for annotation
        annotated_frame = np.copy(frame)

        # Verify if any of the two templates were found in the current frame
        for i, locations in enumerate(beadLocations):
            if current_frame in dataFrameNumbers[i]:

                index = dataFrameNumbers[i].index(current_frame)

                # IF so, annotate the frame with that found template.
                annotated_frame = annotate_frame(annotated_frame, origin_coordinates[i], locations[index], fineBeadLocations[i][index], ROI_coordinates[i], i == rightMost_ROI)

        # Write annotated frame (frames where no templates were found will have no marks)
        out.write(annotated_frame)

        # Complete an analysis step each time a frame is analyzed.
        progress_info["completed_steps"] += 1

        current_steps = progress_info["current_steps"]
        current_total_steps = progress_info["current_total_steps"]
        completed_steps = progress_info["completed_steps"]
        total_steps = progress_info["total_steps"]
        videos_count = progress_info["videos_count"]

        progress_info["analyzis_progression"] = completed_steps/total_steps

        current_progressions = progress_info["current_progressions"]
        analyzis_progression = progress_info["analyzis_progression"]
        analyzed_videos = progress_info["analyzed_videos"]
        current_task = progress_info["current_task"]

        progress_info["progress_description"] = f"Annotating Video {analyzed_videos} of {videos_count} | Current Task: {current_task} | Total Progression: {analyzis_progression*100:.1f}%"

        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            if current_frame < frame_count:
                Error = f"Error: Unable to read the next frame when annotating the video"
                progress_info["progress_description"] = Error
            break
    
    # Release VideoWriter object.
    out.release()

def plot_data(data, frameRate, output_path):

    """
    Plots displacement data over time and saves the plot as an image file.

    Parameters:
    ----------
    data : list of tuples
        A list of tuples where each tuple contains a frame number and the corresponding displacement value.
    frameRate : float
        The frame rate of the video, used to convert frame numbers to seconds.
    output_path : str
        The path where the plot image will be saved.

    Returns:
    -------
    None
        The function generates a plot and saves it as an image file but does not return any value.

    Notes:
    -----
    - The function smooths the displacement data using a moving average before plotting.
    - Converts frame numbers to seconds for the x-axis of the plot.

    """
    # Extract frames and displacements for plotting
    frames = np.array([item[0] for item in data])
    displacements = np.array([item[1] for item in data])

    # Smooth out the data
    window_size = 5
    displacements = np.convolve(displacements, np.ones(window_size)/window_size, mode='same')

    # Convert frames data to seconds
    seconds = frames / frameRate

    # Plot
    plt.figure()
    plt.plot(seconds, displacements)
    
    # Add labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Displacement (μm)')
    plt.legend()

    # Save the plot as a figure
    plt.savefig(output_path)

def annotate_frame(frame, origin_coordinates, current_coordinates, fine_coordinates, roi, rightROI):
    
    """
    Annotates a video frame with the origin and current bead locations.

    Parameters:
    ----------
    frame : np.ndarray
        The video frame to be annotated.
    origin_coordinates : tuple
        The (row, column) coordinates of the bead's original location.
    current_coordinates : tuple
        The (row, column) coordinates of the bead's current location.
    fine_coordinates : tuple
        The fine-tuned (row, column) coordinates of the bead's current location.
    roi : tuple
        A tuple containing the coordinates and dimensions of the region of interest (ROI) (row, column, width, height).
    rightROI : bool
        A flag indicating if this ROI is the rightmost ROI.

    Returns:
    -------
    frame : np.ndarray
        The annotated video frame.

    Notes:
    -----
    - The function centers the origin and current coordinates with respect to the center of the ROI.
    - Draws circles on the frame to indicate the original and current positions of the bead.
    - Adds text annotations for the fine-tuned coordinates near the current position.
    - Adjusts the position of the text annotation based on whether the ROI is the rightmost one.

    """
    _, _, roi_width, roi_height = roi

    # Center Origin Coordinates with center of ROI
    adjusted_origin_coordinates_x = origin_coordinates[0] + roi_width  // 2 # ROW
    adjusted_origin_coordinates_y = origin_coordinates[1] + roi_height // 2 # COLUMN

    # Center current position with center of ROI
    adjusted_object_x = current_coordinates[0] + roi_width  // 2 # ROW
    adjusted_object_y = current_coordinates[1] + roi_height // 2 # COLUMN


    # Draw origin on the frame
    cv2.circle(frame, (int(adjusted_origin_coordinates_y), int(adjusted_origin_coordinates_x)), 2, (255, 0, 0), -1)  # Blue dot for original position
    
    # Draw dots on the frame
    cv2.circle(frame, (int(adjusted_object_y), int(adjusted_object_x)), 1, (0, 0, 255), -1)  # Red dot for current position
    
    # Display object's position in output frame
    object_position_text = f"({fine_coordinates[1]},{fine_coordinates[0]})"
    
    # Add coordinate texts to the frame
    if rightROI:
        cv2.putText(frame, object_position_text, (int(adjusted_object_y) + 10, int(adjusted_object_x) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.putText(frame, object_position_text, (int(adjusted_object_y) -100, int(adjusted_object_x) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame

def process_image(frame):

    """
    Processes a video frame to enhance features for template matching.

    Parameters:
    ----------
    frame : np.ndarray
        The input video frame to be processed.

    Returns:
    -------
    processed_frame : np.ndarray
        The processed frame, enhanced and binarized for template matching.

    Notes:
    -----
    The function performs the following steps:
      1. Converts the frame to grayscale.
      2. Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) for normalization.
      3. Denoises the image using a bilateral filter.
      4. Applies adaptive thresholding to binarize the image.
      5. Erodes the image to reduce noise further.

    """

    # Create parameters for claheing
    clip_limit=20
    tile_grid_size=(4, 4)

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Convert to grayscale
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # Apply clahe normalization
    processed_frame = clahe.apply(processed_frame) 

    # Denoise the image
    processed_frame = cv2.bilateralFilter(processed_frame, 9, 75, 75)

    # Threshold the frame
    processed_frame = cv2.adaptiveThreshold(processed_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 8)

    # Erode the frame (further decfease the noise)
    kernel = np.ones((2, 2), np.uint8) 
    processed_frame = cv2.erode(processed_frame, kernel)

    return processed_frame

def resample_surface(ccorr_matrix, bestMatchLoc, rangeUse=1, resIncrease=100):

    """
    Refines the location of the best match in the cross-correlation matrix using cubic interpolation.

    Parameters:
    ----------
    ccorr_matrix : np.ndarray
        The cross-correlation matrix obtained from template matching.
    bestMatchLoc : tuple
        The (row, column) coordinates of the initial best match location in the cross-correlation matrix.
    rangeUse : int, optional
        The number of pixels around the best match location to use for interpolation (default is 1).
    resIncrease : int, optional
        The factor by which to increase the resolution of the interpolation grid (default is 100).

    Returns:
    -------
    fineLoc : tuple
        The refined (row, column) coordinates of the best match location in the cross-correlation matrix.
    cubic_interpolation : np.ndarray
        The interpolated cross-correlation matrix around the best match location.

    Raises:
    ------
    ValueError
        If `rangeUse` is not a positive integer.

    Notes:
    -----
    - The function extracts a sub-array around the best match location, performs cubic interpolation to refine the match location, and then transforms the interpolated coordinates back to the original cross-correlation matrix coordinate system.
    
    """

    if not isinstance(rangeUse, int) or rangeUse <= 0:
        raise ValueError("rangeUse must be a positive integer")

    # Calculate the bounds for the sub-array extraction
    # bestMatchLoc is in (ROW, COLUMN) coordinate system
    row_min = max(0, bestMatchLoc[0] - rangeUse)
    row_max = min(ccorr_matrix.shape[0], bestMatchLoc[0] + rangeUse + 1) # +1 to include the end index
    column_min = max(0, bestMatchLoc[1] - rangeUse)
    column_max = min(ccorr_matrix.shape[1], bestMatchLoc[1] + rangeUse + 1) # +1 to include the end index

    # Extract data around the peak within rangeUse pixels away
    bestMatch_areaData = ccorr_matrix[row_min:row_max, column_min:column_max]

    # Adjust grid sizes based on actual data extracted
    grid_y, grid_x = np.mgrid[0:1:complex(0, bestMatch_areaData.shape[0]), 0:1:complex(0, bestMatch_areaData.shape[1])]

    # Define finer grid for interpolation
    fine_y, fine_x = np.mgrid[0:1:complex(0, resIncrease), 0:1:complex(0, resIncrease)]

    # Interpolate using cubic interpolation
    cubic_interpolation = griddata((grid_x.ravel(), grid_y.ravel()), bestMatch_areaData.ravel(),
                                   (fine_x, fine_y), method='cubic')

    # Find the coordinates of the maximum in the interpolated data
    max_index = np.unravel_index(np.argmax(cubic_interpolation, axis=None), cubic_interpolation.shape)
    max_index = (max_index[0] / resIncrease, max_index[1] / resIncrease)

    # Transform interpolated coordinates back to the ccorr_matrix coordinate system
    fineLoc = (max_index[0] - 0.5, max_index[1] - 0.5)
    fineLoc = (bestMatchLoc[0] + fineLoc[0], bestMatchLoc[1] + fineLoc[1])

    return fineLoc, cubic_interpolation

def find_best_match(ccorr_matrix, previous_peak, ccorr_threshold, dist_threshold):
    
    """
    Finds the best match locations in the cross-correlation matrix based on a correlation threshold and distance threshold.

    Parameters:
    ----------
    ccorr_matrix : np.ndarray
        The cross-correlation matrix obtained from template matching.
    previous_peak : tuple
        The (row, column) coordinates of the previous best match location.
    ccorr_threshold : float
        The minimum correlation value threshold for considering a match.
    dist_threshold : float
        The maximum distance allowed between the previous peak and the current match location.

    Returns:
    -------
    sorted_ccorr_loc_pairs : list of tuples
        A list of tuples where each tuple contains a correlation value and its corresponding (row, column) location, sorted in descending order by correlation value.
    matchIndex : int
        The index of the best match location in the sorted list that meets the distance threshold. Returns -1 if no match is found within the distance threshold.

    Notes:
    -----
    - The function first finds all locations in the cross-correlation matrix where the correlation value exceeds the threshold.
    - These locations are then sorted by correlation value in descending order.
    - The function then iteratively checks the distance of each location from the previous peak until a match within the distance threshold is found or the list is exhausted.
    
    """
    # Find locations where ccorr value is above the ccorr threshold
    matchLocations = np.where(ccorr_matrix >= ccorr_threshold) # tuple of arrays, where each array represents the indices along one dimension (row and column) where the condition is true

    # If no ccorr value is higher than the threshold, return an empty list.
    if len(matchLocations) == 0:
        sorted_ccorr_loc_pairs = []
        matchIndex = -1
        return sorted_ccorr_loc_pairs, matchIndex

    # Extract correlation value and their corresponding locations
    matchCcorrValues = ccorr_matrix[matchLocations]
    matchLocations = list(zip(*matchLocations)) # List of locations (ROW, COLUMN)

    # Combine the correlation values and locations
    ccorr_loc_pairs = list(zip(matchCcorrValues, matchLocations)) # list of (CCORR, (ROW, COLUMN))

    # Sort ccorr_loc_pairs based on correlation value in descending order.
    sorted_ccorr_loc_pairs = sorted(ccorr_loc_pairs, key = lambda x: x[0], reverse = True)

    # Initialize distance from match in previous frame at a very high number
    matchDist = 1000
    matchIndex = 0 # Index of highest ccorr_value
    
    while matchDist > dist_threshold:
        
        # If out of bounds, break the loop
        if matchIndex > len(sorted_ccorr_loc_pairs) - 1:
            matchIndex = -1 # Return a -1 match index if no item with ccorr_value is located close enough to previous peak location.
            break

        # Calculate distance between previous peak and current bestMatch
        current_match = sorted_ccorr_loc_pairs[matchIndex]
        matchDist = np.sqrt((current_match[1][0] - previous_peak[0])**2 + (current_match[1][1] - previous_peak[1])**2)
        
        # Break the loop if a good match is found.
        if matchDist <= dist_threshold:
            break

        # Read next item in sorted_ccorr_loc_pairs list
        matchIndex += 1

    return sorted_ccorr_loc_pairs, matchIndex

def calculateDistances(location_data):

    """
    Calculates the displacement of each bead from its origin for each frame in the location data.

    Parameters:
    ----------
    location_data : list of list of tuples
        A nested list where each sublist contains tuples of (row, column) coordinates representing bead locations across frames.

    Returns:
    -------
    displacementData : list of list of float
        A nested list where each sublist contains the displacement values of the bead from its origin for each frame.

    Notes:
    -----
    - The function calculates the Euclidean distance between each location and the origin (first location in the list).

    """

    displacementData = [[] for _ in location_data]
    
    for i , list in enumerate(location_data):
        
        origin = list[0]

        for location in list:
            
            displacement = np.sqrt((location[0]-origin[0])**2 + (location[1]-origin[1])**2)
            displacementData[i].append(displacement)
    
    return displacementData

def write_to_CSV(data, output_path):

    """
    Writes the given data to a CSV file at the specified output path.

    Parameters:
    ----------
    data : list of tuples
        A list of tuples where each tuple contains a frame number and either coordinates (x, y) or a displacement value.
    output_path : str
        The path where the CSV file will be saved.

    Returns:
    -------
    None
        The function writes data to a CSV file but does not return any value.

    Notes:
    -----
    - If the second element in the tuples is a tuple, the data is assumed to be frame coordinates and the CSV will have columns for frame number, x coordinate, and y coordinate.
    - Otherwise, the data is assumed to be displacements and the CSV will have columns for frame number and displacement.

    """
        
    # Write data into CSV
    with open(output_path, "w", newline = "") as csvfile:
        
        writer = csv.writer(csvfile)
        
        if isinstance(data[0][1], tuple): # If the data corresponds to frame coordinates.
            
            # Write the header
            writer.writerow(['Frame Number', 'X Coordinate', 'Y Coordinate'])

            # Write data rows
            for frame_number, (x_coord, y_coord) in data:
                 
                 writer.writerow([frame_number, x_coord, y_coord])
        
        else:
            
            # Write header
            writer.writerow(["Frame Number", "Displacement (uM)"])

            # Write data rows
            for frame_number, displacement in data:
                
                writer.writerow([frame_number, displacement])