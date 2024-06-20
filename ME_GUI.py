import customtkinter
import callback_functions as ff

progress_info = {
    "analysis_button_pressed": True, # Did the analysis button get pressed?
    "Analyzing":False,               # Is the analysis currently running (turns off if analysis button is pressed)
    "videos_count": 0,               # How many videos are being analyzed?
    "total_steps": 0,                # Total analysis steps with all of the videos counted
    "analyzed_videos":0,             # Number of videos that were analysed
    "completed_steps": 0,            # Number of steps that were completed
    "current_steps": 0,              # Number of compelted analysis steps in the video currently being analyzed
    "current_total_steps":0,         # Total number of steps in the analysis of the current video
    "current_progressions":0,        # Progression of the analysis of the video currently being analysed
    "analyzis_progression": 0,       # Progresison of the total analysis
    "current_task":"",               # Description of the task currently being performed
    "progress_description": ""       # Description of the total progress in the analysis
    }

class MainApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # Configure Window
        self.title("Molecular Mechanics Analysis")
        self.geometry("1275x700")

        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight = 1)

        self.analyzing = False
        
        # Set default Parameters
        self.extend = 10
        self.ccorr_threshold = 0.4
        self.dist_threshold = 2
        self.manual_ROI_dim = 23
        self.annotated_videos = True
        self.saveTemplates = False
        self.saveCCORR = False
        self.saveAnalysisFrames = False
        self.savePlots = True

        #Thinking icon
        self.thinking_icon = ["|", "/", "-", "\\"]
        self.current_icon = 0
        self.update_delay = 20
        self.number_of_updates = 0

        # Settings frame
        self.settings = settings_frame(self)
        self.settings.grid(row=0, column=0, rowspan = 3, sticky="news")

        # Video Selection, Navigation, and Display frame
        self.video_analysis_frame = video_analysis_frame(self)
        self.video_analysis_frame.grid(row=0, column=1, columnspan =2, padx= (15, 10), pady = 10, sticky="ew")

        # Anlysis Progress Bar
        self.progress_description = customtkinter.CTkLabel(self, text = "Analysis Progression")
        self.progress_description.grid(row=1, column=1, sticky="w", padx=15, pady = 0)        
        self.progress_all = customtkinter.CTkProgressBar(self, orientation="horizontal")
        self.progress_all.grid(row=2,column=1, sticky="ew", padx=(15,5), pady=(0, 15))
        self.progress_all.set(1)

        # Analysis Cancel Button
        self.analysis_button = customtkinter.CTkButton(self, text= "ANALYZE", fg_color="#78716C", hover_color="#A8A29E", 
                                                       command= lambda: ff.analyzis_button_callback(self, progress_info))
        self.analysis_button.grid(row=1, rowspan = 2, column=2, padx = 10, pady = (0, 10), sticky="ew")

    def update_gui(self):
        
        global progress_info
        
        if progress_info["Analyzing"] == True:

            if progress_info["analysis_button_pressed"] == True:
                progress_info["analysis_button_pressed"] = False     

                # Adjust widgets in GUI 
                self.analysis_button.configure(text  = "CANCEL", fg_color = "#EF4444", hover_color = "#F87171")
                self.video_analysis_frame.destroy()
                self.analyzing_label = customtkinter.CTkLabel(self, text = f"ANALYZING {self.thinking_icon[self.current_icon]}")
                self.analyzing_label.grid(row=0, column=1, columnspan =2, padx= (15, 10), pady = 10, sticky="ew")          
                settings_widgets = self.settings.winfo_children()
                
                # Disable all widgets in settings
                for widget in settings_widgets:
                    if isinstance(widget, customtkinter.CTkButton):
                    
                        widget.configure(state="disabled", fg_color="#616161")
                    
                    elif isinstance(widget, customtkinter.CTkLabel) and widget not in [self.settings.title, self.settings.watermark]:
                        widget.configure(text_color="#616161")

                    elif isinstance(widget, customtkinter.CTkSwitch):
                        widget.configure(state="disabled", progress_color="#616161")

                    elif isinstance(widget, customtkinter.CTkOptionMenu):
                        widget.configure(state="disabled")
            
            # Update analysis information and progress bar
            self.progress_all.set(progress_info["analyzis_progression"])
            self.progress_description.configure(text=progress_info["progress_description"] + self.thinking_icon[self.current_icon])
            self.analyzing_label.configure(text = f"ANALYZING {self.thinking_icon[self.current_icon]}")

            if self.current_icon < len(self.thinking_icon) - 1:
                self.current_icon += 1
            else:
                self.current_icon = 0
            
            if self.number_of_updates < self.update_delay:
                self.number_of_updates += 1
            else:
                self.number_of_updates = 0
            
            # Schedule this function to run again after 100 milliseconds
            self.after(100, self.update_gui)

        else:
            self.analysis_button.configure(text = "ANALYZE", fg_color="#78716C", hover_color="#A8A29E")
            self.progress_all.set(1)
            self.progress_description.configure(text=progress_info["progress_description"])
            self.video_analysis_frame = video_analysis_frame(self)
            self.video_analysis_frame.grid(row=0, column=1, columnspan =2, padx= (15, 10), pady = 10, sticky="ew")
            settings_widgets = self.settings.winfo_children()
            
            for widget in settings_widgets:
                if isinstance(widget, customtkinter.CTkButton):
                    widget.configure(state="normal", fg_color="#78716C")
                
                elif isinstance(widget, customtkinter.CTkLabel):
                    widget.configure(text_color="#FAFAFA")

                elif isinstance(widget, customtkinter.CTkSwitch):
                    widget.configure(state="normal", progress_color="#1565C0")

                elif isinstance(widget, customtkinter.CTkOptionMenu):
                    widget.configure(state="normal")

            progress_info = {
                "analysis_button_pressed": True,
                "Analyzing":False, 
                "videos_count": 0, 
                "total_steps": 0, 
                "analyzed_videos":0, 
                "completed_steps": 0, 
                "current_steps": 0,
                "current_total_steps":0, 
                "current_progressions":0, 
                "analyzis_progression": 0,
                "current_task":"",
                "progress_description": ""
                }

class settings_frame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_rowconfigure(16, weight=1)

        # Frame Title
        self.title = customtkinter.CTkLabel(self, text="Laser Trap Wizard", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.title.grid(row=0, column=0, columnspan = 5, padx = 10, pady = (10, 0), sticky = "sew")

        # Template Macthing Seetings Title
        self.template_mathching_title = customtkinter.CTkLabel(self, text = "Template Matching", font = ("roboto", 14, "bold"))
        self.template_mathching_title.grid(row=1, column=0, padx=(10,0), pady=(10,0), sticky="w")

        # Matching Region Extend = INT defining how many pixels larger is the matching region  (in ba.template_matching)
        self.matching_region_extend_label =  customtkinter.CTkLabel(self, text = "Match Region Extend", font = ("roboto", 12) )
        self.matching_region_extend_label.grid(row=3, column =0, padx=(15, 0), pady=(10,0), sticky = "w")
        self.matching_region_extend_decrease = customtkinter.CTkButton(self, text = "-", fg_color="#78716C", hover_color="#A8A29E", width = 10, command = lambda: ff.decreaseButton_callback(master, "extend", self.matching_region_extend_display))
        self.matching_region_extend_decrease.grid(row=3, column=1, padx=(3,0), pady = (10,0))
        self.matching_region_extend_display = customtkinter.CTkLabel(self, text = f"{master.extend}", fg_color="#212121", width = 70)
        self.matching_region_extend_display.grid(row=3, column = 2, pady= (10, 0),sticky = "we")
        self.matching_region_extend_increase = customtkinter.CTkButton(self, text = "+", fg_color="#78716C", hover_color="#A8A29E", width = 10, command = lambda: ff.increaseButton_callback(master, "extend", self.matching_region_extend_display))
        self.matching_region_extend_increase.grid(row=3, column=3, pady = (10,0), padx = (0, 15))

        # CCORR Threshold = Float defining the threshold cross-correlation value to consider the object as "found"
        self.ccorr_threshold_label =  customtkinter.CTkLabel(self, text = "CCORR Threshold", font = ("roboto", 12) )
        self.ccorr_threshold_label.grid(row=4, column =0, padx=(15, 0), pady=(10,0), sticky = "w")
        self.ccorr_threshold_decrease = customtkinter.CTkButton(self, text = "-", fg_color="#78716C", hover_color="#A8A29E", width = 10, command = lambda: ff.decreaseButton_callback(master, "ccorr_threshold", self.cccorr_threshold_display))
        self.ccorr_threshold_decrease.grid(row=4, column=1, padx=(3,0), pady = (10,0))
        self.cccorr_threshold_display = customtkinter.CTkLabel(self, text = f"{master.ccorr_threshold}", fg_color="#212121", width = 70)
        self.cccorr_threshold_display.grid(row=4, column = 2, pady= (10, 0),sticky = "we")
        self.ccorr_threshold_increase = customtkinter.CTkButton(self, text = "+", fg_color="#78716C", hover_color="#A8A29E", width = 10, command = lambda: ff.increaseButton_callback(master, "ccorr_threshold", self.cccorr_threshold_display))
        self.ccorr_threshold_increase.grid(row=4, column=3, pady = (10,0), padx = (0, 15))

        # DISTANCE Threshold = Float defining the threshold distance value to consider the object as "found"
        self.dist_threshold_label =  customtkinter.CTkLabel(self, text = "Distance Threshold", font = ("roboto", 12))
        self.dist_threshold_label.grid(row=5, column =0, padx=(15, 0), pady=(10,0), sticky = "w")
        self.dist_threshold_decrease = customtkinter.CTkButton(self, text = "-", fg_color="#78716C", hover_color="#A8A29E", width = 10, command = lambda: ff.decreaseButton_callback(master, "dist_threshold", self.dist_threshold_display))
        self.dist_threshold_decrease.grid(row=5, column=1, padx=(3,0), pady = (10,0))
        self.dist_threshold_display = customtkinter.CTkLabel(self, text = f"{master.dist_threshold}", fg_color="#212121", width = 70)
        self.dist_threshold_display.grid(row=5, column = 2, pady= (10, 0),sticky = "we")
        self.dist_threshold_increase = customtkinter.CTkButton(self, text = "+", fg_color="#78716C", hover_color="#A8A29E", width = 10, command = lambda: ff.increaseButton_callback(master, "dist_threshold", self.dist_threshold_display))
        self.dist_threshold_increase.grid(row=5, column=3, pady = (10,0), padx = (0, 15))

        # ROI Creation Title
        self.roi_creation_title = customtkinter.CTkLabel(self, text = "ROI Creation", font = ("roboto", 14, "bold"))
        self.roi_creation_title.grid(row=6, column=0, padx=(10,0), pady=(25,0), sticky="w")

        # Manual ROI dimensions
        self.manual_ROI_dim_label =  customtkinter.CTkLabel(self, text = "ROI dimensions", font = ("roboto", 12) )
        self.manual_ROI_dim_label.grid(row=7, column =0, padx=(15, 0), pady=(10,0), sticky = "w")
        self.manual_ROI_dim_decrease = customtkinter.CTkButton(self, text = "-", fg_color="#78716C", hover_color="#A8A29E", width = 10, command = lambda: ff.decreaseButton_callback(master, "manual_ROI_dim", self.manual_ROI_dim_display))
        self.manual_ROI_dim_decrease.grid(row=7, column=1, padx=(3,0), pady = (10,0))
        self.manual_ROI_dim_display = customtkinter.CTkLabel(self, text = f"{master.manual_ROI_dim}", fg_color="#212121", width = 70)
        self.manual_ROI_dim_display.grid(row=7, column = 2, pady= (10, 0), sticky = "we")
        self.manual_ROI_dim_increase = customtkinter.CTkButton(self, text = "+", fg_color="#78716C", hover_color="#A8A29E", width = 10, command = lambda: ff.increaseButton_callback(master, "manual_ROI_dim", self.manual_ROI_dim_display))
        self.manual_ROI_dim_increase.grid(row=7, column=3, pady = (10,0), padx = (0, 15))        

        # Output Setttings Title
        self.output_setting_title = customtkinter.CTkLabel(self, text = "Ouputs to Save", font = ("roboto", 14, "bold"))
        self.output_setting_title.grid(row=10, column=0, padx=(10, 0), pady=(25,0), sticky="w")

        # Plots
        self.savePlots = customtkinter.CTkSwitch(self, text = "Displacement Plots", font = ("Roboto", 12), command = lambda : ff.saveTemplates_callback(self))
        self.savePlots.grid(row=11, column=0, columnspan = 4, padx=(15,0), pady=(10,0), sticky = "we")
        self.savePlots.select()

        # Annotated Videos
        self.annotated_videos = customtkinter.CTkSwitch(self, text = "Annotated Videos", font = ("Roboto", 12), command = lambda: ff.annotatedVideos_callback(self))
        self.annotated_videos.grid(row=12, column=0, columnspan = 4, padx=(15,0), pady=(10,0), sticky = "we")
        self.annotated_videos.select()

        # Save Templates
        self.saveTemplates = customtkinter.CTkSwitch(self, text = "Templates", font = ("Roboto", 12), command = lambda : ff.saveTemplates_callback(self))
        self.saveTemplates.grid(row=13, column=0, columnspan = 4, padx=(15,0), pady=(10,0), sticky = "we")

        # Save Analysis Frames
        self.saveAnalysisFrames = customtkinter.CTkSwitch(self, text = "Analysis Frames", font = ("Roboto", 12), command = lambda : ff.saveAnalysisFrames_callback(self))
        self.saveAnalysisFrames.grid(row=14, column=0, columnspan = 4, padx=(15,0), pady=(10,0), sticky = "we")

        # Save CCORR
        self.saveCCORR = customtkinter.CTkSwitch(self, text = "CCORR Frames", font = ("Roboto", 12), command = lambda : ff.saveCCORR_callback(self))
        self.saveCCORR.grid(row=15, column=0, columnspan = 4, padx=(15,0), pady=(10,0), sticky = "we")

        # Reset auto-bead detection parameters button
        self.reset_auto_detection_params = customtkinter.CTkButton(self, text = "Reset Settings", fg_color="#78716C", hover_color="#A8A29E", command = lambda: ff.reset_settings(self))
        self.reset_auto_detection_params.grid(row=16, column = 0, columnspan = 4, pady = (20,10))

        # Watermark
        self.watermark = customtkinter.CTkLabel(self, text = "Made by Matheus Schultz", font = ("roboto", 11))
        self.watermark.bind("<Button-1>", lambda event: ff.watermark_callback("http://www.matheusschultz.com"))
        self.watermark.grid(row=17, column=0)

class video_analysis_frame(customtkinter.CTkFrame):

    def __init__(self, master):
        super().__init__(master)

        # Initialize variables for vide_analysis
        self.video_directories = () # Tuple with video directories as strings
        self.video_loaded = {} # Dictionary remembering if videos have been previously loaded or not
        self.rois = {} # Dictionary. Keys = VideoDir, Values = list of tuples with ROI coordinates and dimanesions
        self.templates = {} # Dictionary. Keys = VideoDir, Values = np array of templates for that VideoDir
        self.matching_areas = {} # Dictionary. Keys = VideoDir, Values = list of tuples with matching area coordinate and dimensions
        self.rightMostROIs = {} # Dictionary. Keys = VideoDir, Value = list of ints with index of the rightMost ROI for ROIs corresponding to each VideoDir.
        self.selected_ROI = None
        self.current_video_index = 0 # Index for navigation across videos.
        self.frame = None # Initialize empty first frame
        self.display_frame = None # Initialize empty displayed frame (frame that will contain drawn ROIs)
        self.template_display_size = 200 # Square size of the template display image.
        self.total_frames = 0

        # Label to display frame
        self.video_display = customtkinter.CTkLabel(self, text = "Select Videos for Analysis", fg_color="#212121", width=int(1*696), height=int(1*520))
        self.video_display.grid(row=0, column = 0, rowspan = 4, columnspan=2, padx=15, pady = (15,0), sticky="ew")
        self.video_display.bind("<Button-1>", lambda event: ff.mouse_left_click(self, event, master.manual_ROI_dim))
        self.video_display.bind("<Button-3>", lambda event: ff.mouse_right_click(self, event))
        
        # Labels to display Templates
        self.template_display_left = customtkinter.CTkLabel(self, text = "Left Template", fg_color="#212121", width=self.template_display_size, height=self.template_display_size)
        self.template_display_left.bind("<Button-1>", lambda event: ff.template_selection(self, event, templateLabel="left"))
        self.template_display_left.grid(row=0, column = 2, columnspan = 4, padx = (0, 10), pady=(5,0), sticky="ew")
        
        self.template_display_right = customtkinter.CTkLabel(self, text = "Right Template", fg_color="#212121", width=self.template_display_size, height=self.template_display_size)
        self.template_display_right.grid(row=2, column = 2, columnspan = 4, padx = (0, 10), pady=(5,0), sticky="ew")
        self.template_display_right.bind("<Button-1>", lambda event: ff.template_selection(self, event, templateLabel="right"))

        #Template adjustement buttons
        self.Templ_AdjButt_Left = customtkinter.CTkButton(self, text = "<", width = 30, fg_color="#78716C", hover_color="#A8A29E", command= lambda: ff.fineAdjust(self, com = "left"))
        self.Templ_AdjButt_Left.grid(row = 4, column = 2, padx=(5, 0), pady=(0, 3), sticky = "swe")
        
        self.Templ_AdjButt_Up = customtkinter.CTkButton(self, text = "^", width = 30, fg_color="#78716C", hover_color="#A8A29E", command= lambda: ff.fineAdjust(self, com = "up"))
        self.Templ_AdjButt_Up.grid(row = 5, column = 2, padx=(5, 0), pady=(0, 3), sticky = "nwe")
        
        self.Templ_AdjButt_Down = customtkinter.CTkButton(self, text = "v", width = 30, fg_color="#78716C", hover_color="#A8A29E", command= lambda: ff.fineAdjust(self, com = "down"))
        self.Templ_AdjButt_Down.grid(row = 5, column = 3, padx=(3, 0), pady=(0, 3), sticky = "nwe")
        
        self.Templ_AdjButt_Right = customtkinter.CTkButton(self, text = ">", width = 30, fg_color="#78716C", hover_color="#A8A29E", command= lambda: ff.fineAdjust(self, com = "right"))
        self.Templ_AdjButt_Right.grid(row = 4, column = 3, padx=(3, 0), pady=(0, 3), sticky = "swe")
        
        self.Templ_AdjButt_Increase = customtkinter.CTkButton(self, text = "+", width = 30, fg_color="#78716C", hover_color="#A8A29E", command= lambda: ff.fineAdjust(self, com = "increase"))
        self.Templ_AdjButt_Increase.grid(row = 4, column = 4, padx=(20, 0), pady=(0, 3), sticky = "swe")
        
        self.Templ_AdjButt_Decrease = customtkinter.CTkButton(self, text = "-", width = 30, fg_color="#78716C", hover_color="#A8A29E", command= lambda: ff.fineAdjust(self, com = "decrease"))
        self.Templ_AdjButt_Decrease.grid(row = 5, column = 4, padx=(20, 0), pady=(0, 20), sticky = "nwe")
        
        # Video Selection Button
        self.select_videos_button = customtkinter.CTkButton(self, text="Select Videos", fg_color="#78716C", hover_color="#A8A29E", command = lambda: ff.select_videos(self, progress_info))
        self.select_videos_button.grid(row=4, column=0, columnspan = 2, padx=15, pady=(10,5), sticky="ew")

        # Selected Videos Navigation Buttons
        self.previous_button = customtkinter.CTkButton(self, text="Previous Video", fg_color="#78716C", hover_color="#A8A29E", command= lambda: ff.previous_video(self))
        self.previous_button.grid(row=5, column=0,  padx=(15, 0), pady=(3,20), sticky="ew") 
        self.next_button = customtkinter.CTkButton(self, text="Next Video", fg_color="#78716C", hover_color="#A8A29E", command= lambda: ff.next_video(self))
        self.next_button.grid(row=5, column=1, padx=(5, 15), pady=(3, 20), sticky="ew") 

if __name__=="__main__":
    
    customtkinter.deactivate_automatic_dpi_awareness()
    app = MainApp()
    app.mainloop()