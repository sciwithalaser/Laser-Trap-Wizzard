# Laser Trapping Wizzard

## Overview
I built this application to facilitate the analysis of laser-trapping experiments videos. That is, this application tracks the position of the trapped microscpheres throughout the video, and computes their displacements for later force analysis. 

Object tracking is performed via template matching, tracking at most 2 objects with 2 seperate templates constructed by the user through a graphical user interface.

The application also allows the user to modify template matching parameters, as well as output options.

The application is composed of three main files:

1. `ME_GUI.py`: Contains the graphical user interface (GUI) for the application.
2. `callback_functions.py`: Handles various callbacks and helper functions.
3. `bead_analysis_functions.py`: Contains functions for bead tracking analysis and related operations.

## Requirements
- Python 3.x
- Required Python libraries:
  - `numpy`
  - `cv2` (OpenCV)
  - `matplotlib`
  - `scipy`
  - `customtkinter`
  - `os`
  - `csv`

## Installation
1. Clone the repository:
   ```bash
   git clone <https://github.com/sciwithalaser/Laser-Trap-Wizzard>
   cd <repository_directory>
   ```

2. Install the required Python libraries:
   ```bash
   pip install numpy opencv-python matplotlib scipy customtkinter os csv
   ```

## Usage
### Running the GUI
To start the application, run the `ME_GUI.py` file:
```bash
python ME_GUI.py
```

### Files Description
#### ME_GUI.py
This file sets up the graphical user interface using `customtkinter`. It allows users to select input videos, set parameters for template matching, and visualize the results.

#### callback_functions.py
Contains helper functions and callbacks used in the GUI. This includes functions for loading videos, updating the interface, and handling user interactions.

#### bead_analysis_functions.py
Includes core functions for bead displacement analysis:
- `find_best_match`: Identifies the best matching location in the cross-correlation matrix return by the templateMatching operation.
- `calculateDistances`: Calculates the displacements of beads based on their coordinates.
- `write_to_CSV`: Writes analysis results to a CSV file.
- `resample_surface`: Performs cubic interpolation on the cross-correlation matrix for subpixel localization of the object.
- `annotate_video`: Creates an annotated video showing the position of the tracked_object.

### Example Workflow
1. **Load Video**: Use the GUI to load the video files for analysis.
2. **Set Parameters**: Set the required parameters for template matching (e.g., threshold values, distance thresholds).
3. **Run Analysis**: Start the analysis to compute bead displacements.
4. **Save Results**: The results, including CSV files and plots, will be saved to the specified output directory.
5. **Annotated Video**: An optional annotated video showing the template matching results can also be generated.