Based on the provided files, here’s a draft for the README file for your template matching software:

---

# Template Matching Software

## Overview
This software performs template matching on video frames to analyze bead displacements. It includes functionalities for cross-correlation, displacement calculation, and visualization of results. The application is composed of three main files:

1. `ME_GUI.py`: Contains the graphical user interface (GUI) for the application.
2. `callback_functions.py`: Handles various callbacks and helper functions.
3. `bead_analysis_functions.py`: Contains functions for bead displacement analysis and related operations.

## Requirements
- Python 3.x
- Required Python libraries:
  - `numpy`
  - `cv2` (OpenCV)
  - `matplotlib`
  - `scipy`
  - `customtkinter`

## Installation
1. Clone the repository:
   ```bash
   git clone <https://github.com/sciwithalaser/Laser-Trap-Wizzard>
   cd <repository_directory>
   ```
2. Install the required Python libraries:
   ```bash
   pip install numpy opencv-python matplotlib scipy customtkinter
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
- `find_best_match`: Identifies the best matching location in the cross-correlation matrix.
- `calculateDistances`: Calculates the displacements of beads based on their coordinates.
- `write_to_CSV`: Writes analysis results to a CSV file.
- `resample_surface`: Performs cubic interpolation on the cross-correlation matrix.
- `annotate_video`: Creates an annotated video showing the analysis results.

### Example Workflow
1. **Load Video**: Use the GUI to load the video file for analysis.
2. **Set Parameters**: Set the required parameters for template matching (e.g., threshold values, distance thresholds).
3. **Run Analysis**: Start the analysis to compute bead displacements.
4. **Save Results**: The results, including CSV files and plots, will be saved to the specified output directory.
5. **Annotated Video**: An optional annotated video showing the template matching results can also be generated.

### Sample Code Snippets
#### Loading a Video and Performing Template Matching
```python
import cv2
import numpy as np
from bead_analysis_functions import find_best_match, calculateDistances

# Load video
cap = cv2.VideoCapture('input_video.mp4')

# Read the first frame
ret, frame = cap.read()

# Define the template
template = cv2.imread('template.png', 0)

# Perform template matching
ccorr_matrix = cv2.matchTemplate(frame, template, cv2.TM_CCORR_NORMED)

# Find the best match location
best_match_loc = find_best_match(ccorr_matrix, previous_peak=(0,0), ccorr_threshold=0.8, dist_threshold=5)

# Calculate distances
distances = calculateDistances([best_match_loc])
```

### Writing Results to CSV
```python
from bead_analysis_functions import write_to_CSV

# Example data
data = [(1, 0.1), (2, 0.2), (3, 0.15), (4, 0.25)]
output_path = 'results.csv'

# Write to CSV
write_to_CSV(data, output_path)
```

## License
This project is licensed under the MIT License.

---

### Note:
1. Replace `<repository_url>` and `<repository_directory>` with the actual URL and directory name of your repository.
2. Ensure the paths and example data used in the snippets are relevant to your actual use cases.

Feel free to modify the README content to better suit your needs and provide more specific instructions or information as required.
