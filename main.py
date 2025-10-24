import numpy as np
import matplotlib.pyplot as plt
from tkinter import Button, Text, Scrollbar, BooleanVar, Checkbutton, OptionMenu, BOTH, LEFT, RIGHT, BOTTOM, Y, END
from tkinter import Frame, Label, filedialog, Toplevel, Tk, Entry, Listbox, messagebox
import os, shutil, sys, ctypes
import tkinter as tk
from scipy.signal import savgol_filter
from scipy.ndimage import minimum_filter, uniform_filter1d
from tkinter import Listbox, TOP 
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.collections
from matplotlib.offsetbox import AnnotationBbox, TextArea
from typing import Union
from scipy.special import voigt_profile
from scipy.optimize import minimize
from joblib import Parallel, delayed
from scipy.stats import qmc

from about import aboutprogram

# This block makes the application DPI-aware on Windows, which is essential
# for rendering correctly on high-resolution displays.
# It MUST be called before any Tkinter windows are created.

try:
    if sys.platform == "win32":
        # Set the process DPI awareness for Windows.
        # A value of 1 corresponds to System DPI-aware.
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        print("DPI awareness successfully set for Windows.")
except Exception as e:
    print(f"Warning: Could not set DPI awareness. The GUI may appear small on high-res displays. Error: {e}")

Tk().withdraw()  # Temporary root for file dialogs
Tk().destroy()  # Cleanup after dialog

fit_lines = []  # Stores tuples of (ax1_line, ax2_line)
fitting_params_window_open = False
auto_analysis_window_open = False
# Settings handling
DEFAULT_STATE_INDEX = 0
SETTINGS_FILE = "settings.txt"
DEFAULT_PEAKLIST_FILE = "Peaklist.txt"
DEFAULT_FWHM = 0.1
DEFAULT_FWHM_G = 0.05 # Default Gaussian component
DEFAULT_FWHM_L = 0.05 # Default Lorentzian component
DEFAULT_FWHM_VAR = 0.0  # Allowed variation in percent
DEFAULT_FWHM_G_VAR = 0.0 # Voigt Gaussian variation
DEFAULT_FWHM_L_VAR = 0.0 # Voigt Lorentzian variation
DEFAULT_EXT_SCALE = 1.0
DEFAULT_CALC_SCALE = 1.0
DEFAULT_AMPLITUDE_VAR = 100.0
DEFAULT_CENTER_DEV = 0.0
DEFAULT_INTENSITY_THRESH = 0.0  # Show all peaks
DEFAULT_REG_PARAM = 0.0  # New default for linear regularization
DEFAULT_ALPHA = 0.0  # Weighting between contour and integral residuals
DEFAULT_MIN_AMP = 0.0  # Adjust the default as needed
DEFAULT_CONTOUR = 'Gaussian' # Default contour type
DEFAULT_NUM_STARTS = 1
DEFAULT_MIN_PEAK_DIST = 0.0
DEFAULT_MARKER_DIAMETER = 8.0
DEFAULT_DISABLE_MPL_CONTROLS = False

event_cids = []
current_hover_annotation = None     # To store the temporary hover annotation object
current_hover_line = None           # To store the temporary vertical 
base_annotation_artists = []        # To store the persistent, rotated annotations
calc_base_annotation_artists = []   # To store persistent annotations for calculated peaks
current_highlighted_calc_line = None # To store the artist for the highlighted calculated line
REG_PARAM = DEFAULT_REG_PARAM  # Global regularization parameter
ALPHA = DEFAULT_ALPHA
MIN_AMP = DEFAULT_MIN_AMP
peaklist_filename = DEFAULT_PEAKLIST_FILE

UPDATE_PLOT_ON_MOVE = False # Global flag for checkbox state
update_on_move_var = None   # To hold the Tkinter BooleanVar for the checkbox
line_plot_artists = {1: None, 2: None} # Dict to store line artists for ax1 (key 1) 

input_window_open = False
settings_window_open = False
spectrum_processing_window_open = False 
current_vline = None
destination_numbers = None

peak_plot_object = None  # Will hold the single plot object for all peaks
peak_plot_object_ax2 = None # Will hold the single plot object for all peaks on ax2
peak_data_store = []     # Will store ['x', 'y', 'fwhm', 'integral', 'qn1', 'qn2', ...] for each peak
calc_peak_artists = [] # List to store artists created by plot_calculated_peaks
peaklist_annotation_artists = [] # List to store annotation artists created by load_existing_peaks
peaklist_annotation_artists_ax2 = [] # List to store annotation artists for ax2

# New global dictionary for storing structured plot settings
# Format: {graph_num: {plot_num: {'type': '...', 'file': '...'}}}
plot_settings = {}

contour_settings = {}

def position_dialog_near_cursor(dialog, width, height, offset=20):
    """
    Positions a Tkinter Toplevel dialog window near the current mouse cursor.

    Args:
        dialog (Toplevel): The dialog window object to position.
        width (int): The desired width of the dialog.
        height (int): The desired height of the dialog.
        offset (int): The pixel offset from the cursor.
    """
    # Attempt to get the parent window, which is needed to query cursor position.
    # The dialog's master is usually the correct parent.
    parent = dialog.master
    try:
        # Fallback to the main matplotlib figure window if the direct parent is not available.
        if not parent:
            fig_manager = plt.get_current_fig_manager()
            parent = fig_manager.window

        # Get the global screen coordinates of the mouse pointer.
        cursor_x = parent.winfo_pointerx()
        cursor_y = parent.winfo_pointery()

        # Set the window's geometry: "widthxheight+x_offset+y_offset"
        dialog.geometry(f"{width}x{height}+{cursor_x + offset}+{cursor_y + offset}")

    except Exception:
        # If we fail to get cursor position for any reason, just center it.
        dialog.geometry(f"{width}x{height}")

    # Bring the window to the front and give it focus.
    dialog.lift()
    dialog.attributes('-topmost', True)
    dialog.after_idle(dialog.attributes, '-topmost', False)
    dialog.focus_force()


def manage_file(file_path, mode):
    """
    Manage a file by either clearing its contents or removing it entirely.

    Parameters:
        file_path (str): Path to the target file.
        mode (str): Action to perform - 'clear' to empty the file, 'remove' to delete it.

    Raises:
        ValueError: If an invalid mode is specified.
        FileNotFoundError: If the file doesn't exist (in remove mode).
        PermissionError: If lacking required file permissions.
    """
    if mode == 'clear':
        # Open in write mode to truncate the file
        with open(file_path, 'w'):
            pass
    elif mode == 'remove':
        os.remove(file_path)
    else:
        raise ValueError("Invalid mode. Use either 'clear' or 'remove'.")

def safe_write_to_file(filepath: str, lines: list[str]):
    """
    Safely writes a list of lines to a file using a temporary file and rename.

    This prevents data corruption by ensuring the original file is only
    replaced after the new content has been successfully written to disk.

    Parameters:
        filepath (str): The final destination path of the file.
        lines (list[str]): The list of string lines to write to the file.
    
    Raises:
        IOError: If any file operation (write, rename) fails.
    """
    # Define the temporary file path
    temp_filepath = filepath + ".tmp"

    try:
        # 1. Write the new content to the temporary file
        with open(temp_filepath, 'w') as f:
            f.writelines(lines)

        # 2. If the write is successful, replace the original file with the temp file.
        #    On most OSes, this is an atomic operation.
        shutil.move(temp_filepath, filepath)
        
        # print(f"Successfully and safely saved {filepath}") # Optional: for debugging

    except Exception as e:
        print(f"CRITICAL ERROR during safe write to {filepath}: {e}")
        # If an error occurs, try to clean up the temporary file if it exists
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception as e_rem:
                print(f"  - Additionally, failed to remove temp file {temp_filepath}: {e_rem}")
        # Re-raise the original exception so the calling function knows something went wrong
        raise IOError(f"Failed to safely write to {filepath}") from e
    
Gen_output = 'Gen_Output.txt'
if not os.path.exists(Gen_output):
    with open(Gen_output, 'w') as f:
        f.write("Beginning of session\n")
else:
    manage_file(Gen_output, "clear")
    with open(Gen_output, 'w') as f:
        f.write("Beginning of session\n")

# Clear fitlim.txt on startup if it exists and isn't empty
if os.path.exists("fitlim.txt"):
    if os.path.getsize("fitlim.txt") > 0:
        open("fitlim.txt", "w").close()  # Truncate file

def calculate_gaussian_value(
    x: Union[float, np.ndarray],
    amplitude: float,
    center: Union[float, np.ndarray],
    fwhm: float
) -> Union[float, np.ndarray]:
    """
    Calculates the Y-value(s) of a Gaussian peak at given X-coordinate(s).

    This function is vectorized and handles broadcasting, allowing 'x' and 'center'
    to be arrays, which is highly efficient for generating multiple contours.

    Parameters:
        x (Union[float, np.ndarray]): 
            The x-coordinate(s) at which to evaluate the Gaussian function.
        amplitude (float): 
            The maximum height of the Gaussian peak.
        center (Union[float, np.ndarray]): 
            The x-coordinate(s) of the center of the peak.
        fwhm (float): 
            The Full-Width at Half-Maximum of the peak. Must be positive.

    Returns:
        Union[float, np.ndarray]: 
            The calculated y-value(s) of the Gaussian contour.
    """
    if fwhm <= 0:
        # In a fitting context, returning an array of zeros is safer than raising an error
        if isinstance(x, np.ndarray):
            return np.zeros_like(x)
        return 0.0

    # Convert FWHM to sigma (standard deviation)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Calculate the value using the standard Gaussian equation.
    # This works for both single values and NumPy arrays.
    exponent = -((x - center)**2) / (2 * sigma**2)
    return amplitude * np.exp(exponent)

def calculate_lorentzian_value(
    x: Union[float, np.ndarray],
    amplitude: float,
    center: Union[float, np.ndarray],
    fwhm: float
) -> Union[float, np.ndarray]:
    """
    Calculates the Y-value(s) of a Lorentzian peak at given X-coordinate(s).

    This function is vectorized and handles broadcasting, allowing 'x' and 'center'
    to be arrays, which is highly efficient for generating multiple contours.

    Parameters:
        x (Union[float, np.ndarray]): 
            The x-coordinate(s) at which to evaluate the Lorentzian function.
        amplitude (float): 
            The maximum height of the Lorentzian peak.
        center (Union[float, np.ndarray]): 
            The x-coordinate(s) of the center of the peak.
        fwhm (float): 
            The Full-Width at Half-Maximum (FWHM) of the peak. Must be positive.

    Returns:
        Union[float, np.ndarray]: 
            The calculated y-value(s) of the Lorentzian contour.
    """
    if fwhm <= 0:
        # A non-positive FWHM is physically meaningless and can cause division by zero.
        # Returning zeros is a safe and predictable behavior.
        if isinstance(x, np.ndarray):
            return np.zeros_like(x)
        return 0.0

    # The half-width at half-maximum (HWHM), often denoted as gamma (γ).
    gamma = fwhm / 2.0

    # Calculate the value using the standard Lorentzian equation.
    # This works for both single values and NumPy arrays due to broadcasting.
    numerator = gamma**2
    denominator = (x - center)**2 + gamma**2
    
    return amplitude * (numerator / denominator)

def calculate_voigt_value(
    x: Union[float, np.ndarray],
    amplitude: float,
    center: Union[float, np.ndarray],
    fwhm_g: float, # <-- Note: fwhm_g
    fwhm_l: float  # <-- Note: fwhm_l
) -> Union[float, np.ndarray]:
    """
    Calculates the Y-value(s) of a Voigt peak.

    This is a wrapper for scipy.special.voigt_profile that uses FWHM parameters.
    A Voigt profile is a convolution of a Gaussian and a Lorentzian.

    Parameters:
        x: The x-coordinate(s) at which to evaluate the function.
        amplitude: The maximum height of the peak.
        center: The x-coordinate of the center of the peak.
        fwhm_g: The Full-Width at Half-Maximum of the Gaussian component.
        fwhm_l: The Full-Width at Half-Maximum of the Lorentzian component.

    Returns:
        The calculated y-value(s) of the Voigt contour.
    """
    if fwhm_g <= 0 or fwhm_l <= 0:
        return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0

    # Convert FWHM to sigma (for Gaussian) and gamma (HWHM for Lorentzian)
    sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
    gamma = fwhm_l / 2.0

    # voigt_profile returns a value normalized to unit area. We need to scale it.
    # The height of the voigt_profile at the center is voigt_profile(0, sigma, gamma).
    # We divide by this height to normalize it to a unit amplitude, then multiply by our desired amplitude.
    scaling_factor = voigt_profile(0, sigma, gamma)
    if scaling_factor < 1e-10: # Avoid division by zero for extremely narrow peaks
        return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0

    return amplitude * (voigt_profile(x - center, sigma, gamma) / scaling_factor)

CONTOUR_FUNCTIONS = {
    'Gaussian': calculate_gaussian_value,
    'Lorentzian': calculate_lorentzian_value,
    'Voigt': calculate_voigt_value,
    # Future contour functions can be added here
}



def clear_all_peak_data():
    """
    Clears all in-memory peak data (coordinates, storage) and removes 
    all associated artists (scatter plots, annotations) from the figures.
    """
    global peak_plot_object, peak_plot_object_ax2, peak_coords, peak_data_store, base_annotation_artists, fig

    # 1. Clear the in-memory data lists
    peak_coords.clear()
    peak_data_store.clear()

    # 2. Safely remove the scatter plot object from the top axis (ax1)
    if peak_plot_object:
        try:
            peak_plot_object.remove()
        except Exception as e:
            print(f"Warning: Could not remove peak plot object from ax1: {e}")
        finally:
            peak_plot_object = None

    # 3. Safely remove the scatter plot object from the bottom axis (ax2)
    if peak_plot_object_ax2:
        try:
            peak_plot_object_ax2.remove()
        except Exception as e:
            print(f"Warning: Could not remove peak plot object from ax2: {e}")
        finally:
            peak_plot_object_ax2 = None

    # 4. Safely remove all annotation artists
    for artist in base_annotation_artists:
        try:
            artist.remove()
        except Exception as e:
            print(f"Warning: Could not remove an annotation artist: {e}")
    base_annotation_artists.clear()

    # 5. Redraw the canvas to reflect the removals
    if fig and fig.canvas:
        fig.canvas.draw_idle()
    
    print("Cleared all existing peak data and removed visual elements from the plot.")

def show_settings_dialog():
    """Create and display the settings editing dialog with switchable groups."""

    global settings_window_open, current_settings, plot_settings, contour_settings, fig, ax1
    global calc_filename, calc_peak_artists, peak_plot_object, peak_coords, peak_data_store
    global update_on_move_var, UPDATE_PLOT_ON_MOVE
    global peaklist_filename
    global DEFAULT_FWHM, FWHM_VAR, EXT_SCALE, CALC_SCALE, INTENSITY_THRESH, CENTER_DEV
    global REG_PARAM, ALPHA, STATE_INDEX, MIN_AMP
    global DEFAULT_FWHM_G, DEFAULT_FWHM_L, DEFAULT_FWHM_G_VAR, DEFAULT_FWHM_L_VAR
    
    if settings_window_open:
        return
    settings_window_open = True

    settings_config = [
        {'key': 'ext_scale', 'label': 'External Data Scale:', 'type': float,
         'validation': lambda v: True, 'tooltip': '# Scaling for external data (lower plot)', 'redraw_trigger': True},
        {'key': 'calc_scale', 'label': 'Calculated Peaks Scale:', 'type': float,
         'validation': lambda v: True, 'tooltip': '# Scaling for calculated peaks (upper plot)', 'redraw_trigger': True},
        {'key': 'marker_diameter', 'label': 'Marker Diameter (pts):', 'type': float,
         'validation': lambda v: v > 0, 'tooltip': '# Visual size of peak markers', 'redraw_trigger': True},
        {'key': 'intensity_thresh', 'label': 'Intensity Threshold:', 'type': float,
         'validation': lambda v: 0.0 <= v <= 1.0, 'tooltip': '# Min relative intensity (0.0-1.0)', 'redraw_trigger': True},
        {'key': 'state_index', 'label': 'State Index:', 'type': 'int_spinner',
         'validation': lambda v: v >= 0, 'tooltip': '# Select state index (non-negative integer)', 'redraw_trigger': False},
    ]

    fig_manager = plt.get_current_fig_manager()
    root = fig_manager.window if fig_manager else tk.Tk()

    dialog = Toplevel(root)
    dialog.title("COROVA main window")
    position_dialog_near_cursor(dialog, width=800, height=600)

    top_frame = Frame(dialog)
    top_frame.pack(side=tk.TOP, fill='x', padx=10, pady=(10, 5))
    content_container = Frame(dialog)
    content_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    bottom_button_frame = Frame(dialog)
    bottom_button_frame.pack(side=tk.BOTTOM, fill='x', padx=10, pady=10)

    content_container.grid_rowconfigure(0, weight=1)
    content_container.grid_columnconfigure(0, weight=1)

    general_frame = Frame(content_container)
    plot_frame = Frame(content_container)
    contour_frame = Frame(content_container)
    controls_frame = Frame(content_container)

    # Add content to the Controls pane ---
    controls_widgets = {} # To store the widget for saving
    controls_frame.columnconfigure(0, weight=1) # Allow content to expand

    disable_mpl_frame = Frame(controls_frame)
    disable_mpl_frame.pack(padx=10, pady=10, fill='x')
    
    # Create a Tkinter BooleanVar to link to the checkbox
    dialog.disable_mpl_var = BooleanVar(master=dialog, value=current_settings.get('disable_mpl_controls', False))
    
    cb = Checkbutton(disable_mpl_frame,
                     text="Disable Matplotlib implicit key controls (e.g., 'f', 's', 'q')",
                     variable=dialog.disable_mpl_var)
    cb.pack(side=LEFT, anchor='w')
    controls_widgets['disable_mpl_controls'] = dialog.disable_mpl_var


    general_frame.grid(row=0, column=0, sticky='nsew')
    plot_frame.grid(row=0, column=0, sticky='nsew')
    contour_frame.grid(row=0, column=0, sticky='nsew')
    controls_frame.grid(row=0, column=0, sticky='nsew')

    plot_button = Button(bottom_button_frame, text="plot", command=lambda: messagebox.showinfo("Plot", "Placeholder.", parent=dialog))

    Label(top_frame, text="Settings Group:").pack(side=tk.LEFT)
    group_var = tk.StringVar(dialog, value="General")
    def switch_group(selected_group):
        plot_button.pack_forget()
        if selected_group == "General": general_frame.tkraise()
        elif selected_group == "Contour": contour_frame.tkraise()
        elif selected_group == "Plot":
            plot_frame.tkraise()
            plot_button.pack(side=LEFT, padx=5)
        elif selected_group == "Controls":
            controls_frame.tkraise()

    OptionMenu(top_frame, group_var, "General", "Contour", "Plot", "Controls", command=switch_group).pack(side=tk.LEFT, padx=5)

    # Operational Module Dropdown ---
    Label(top_frame, text="Operational Module:").pack(side=tk.LEFT, padx=(20, 5)) # Add space before the label
    
    module_var = tk.StringVar(dialog)
    module_var.set("...") # Set the default placeholder text

    def on_module_select(selected_module):
        """Called when a user selects an item from the Operational Module menu."""
        if selected_module == "Manual rovibrational analysis":
            # This logic calls the Interpretation function.
            close_dialog_and_flag()
            Interpretation(calc_filename)
        # logic for Baseline correction ---
        elif selected_module == "Spectrum processing":
            close_dialog_and_flag()
            show_spectrum_processing_dialog()
        # logic for the automatic rotational vibrational analysis
        elif selected_module == "Automatic rovibrational analysis":
            close_dialog_and_flag()
            automatic_rovibrational_analysis()
        # Reset the dropdown to the placeholder after the action is performed
        module_var.set("...")

    # The OptionMenu itself
    module_menu = OptionMenu(top_frame, module_var, "none", "Manual rovibrational analysis", "Spectrum processing", "Automatic rovibrational analysis", command=on_module_select)
    module_menu.pack(side=tk.LEFT)

    # ===================================================================
    # --- GENERAL SETTINGS PANE ---
    # ===================================================================
    general_settings_widgets = {}

    # --- NEW: Peaklist File Selection ---
    peaklist_frame = Frame(general_frame)
    peaklist_frame.pack(padx=10, pady=5, fill='x')
    
    # Use a StringVar to easily update the label
    dialog.peaklist_var = tk.StringVar(master=dialog, value=peaklist_filename)

    def _select_peaklist_file():
        """Opens a file dialog to select a new peaklist file."""
        filepath = filedialog.askopenfilename(
            title="Select Peaklist Database File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=os.path.dirname(peaklist_filename), # Start in the current file's directory
            parent=dialog
        )
        if filepath:
            # Update the StringVar, which automatically updates the label's text
            dialog.peaklist_var.set(filepath)
            print(f"Peaklist file selected: {filepath}")

    Button(peaklist_frame, text="Choose DB File", command=_select_peaklist_file).pack(side=LEFT, padx=(0, 5))
    # This label shows the currently selected file path
    Label(peaklist_frame, textvariable=dialog.peaklist_var, relief="sunken", anchor='w', wraplength=500).pack(side=LEFT, fill='x', expand=True)

    for config in settings_config:
        key = config['key']
        frame = Frame(general_frame)
        frame.pack(padx=10, pady=5, fill='x')
        Label(frame, text=config['label'], width=22, anchor='w').pack(side='left')
        current_val = current_settings.get(key)
        if config['type'] == 'int_spinner':
            spinner_frame = Frame(frame)
            spinner_frame.pack(side='left', fill='x', expand=True)
            val_label = Label(spinner_frame, text=str(current_val), width=8, relief="sunken", anchor='e')
            val_label.pack(side='left', padx=(0, 5))
            general_settings_widgets[key] = {'display_label': val_label, 'current_value': current_val}
            def _update_spinner(delta, k_local=key):
                 widget_info = general_settings_widgets[k_local]
                 config_local = next(c for c in settings_config if c['key'] == k_local)
                 new_value = widget_info['current_value'] + delta
                 if config_local['validation'](new_value):
                     widget_info['current_value'] = new_value
                     widget_info['display_label'].config(text=str(new_value))
                 else: dialog.bell()
            Button(spinner_frame, text="+", command=lambda k=key: _update_spinner(1, k), width=2).pack(side='left')
            Button(spinner_frame, text="-", command=lambda k=key: _update_spinner(-1, k), width=2).pack(side='left')
        else:
            entry = Entry(frame)
            if current_val is not None: entry.insert(0, str(current_val))
            entry.pack(side='left', fill='x', expand=True)
            general_settings_widgets[key] = {'entry': entry}
        Label(frame, text=config['tooltip'], fg='gray', font=('Arial', 8)).pack(side='left', padx=5)

    # General Action buttons
    general_action_button_frame = Frame(general_frame)
    general_action_button_frame.pack(pady=10, fill='x', padx=10)

    def show_manual_window():
        """Creates and displays a new, advanced manual window with topics."""
        manual_window = Toplevel(dialog)
        manual_window.title("About program")
        position_dialog_near_cursor(manual_window, width=850, height=550)

        # Load the logo image ---
        # We load the image once and store a reference to it on the window object
        # to prevent it from being garbage collected.
        try:
            # THE FIX: Explicitly set the 'master' of the image to the window
            # it will be displayed in. This correctly links its lifecycle.
            logo_photo = tk.PhotoImage(master=manual_window, file="logo.gif")
            manual_window.logo_image = logo_photo  # We still keep the reference, which is best practice.
        except tk.TclError:
            print("Warning: Could not find 'logo.gif'. The about window will be displayed without the logo.")
            manual_window.logo_image = None # Explicitly set to None if it fails

        # --- Window Layout ---
        # Main frame to hold the two sub-frames
        main_frame = Frame(manual_window)
        main_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Left frame for buttons
        left_frame = Frame(main_frame) 
        left_frame.pack(side=LEFT, fill=Y, padx=(0, 5))

        # Right frame for the text widget
        right_frame = Frame(main_frame)
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        # --- Text Widget and Scrollbar ---
        scrollbar = Scrollbar(right_frame)
        text_widget = Text(right_frame, wrap='word', yscrollcommand=scrollbar.set, font=("Arial", 10), spacing1=2, spacing3=5)
        scrollbar.config(command=text_widget.yview)

        scrollbar.pack(side=RIGHT, fill=Y)
        text_widget.pack(side=LEFT, fill=BOTH, expand=True)

        # --- Function to update text display ---
        def update_text(topic):
            # Get the help text from the dictionary
            content = aboutprogram.get(topic, "Topic not found.")

            # Enable the text widget to modify it
            text_widget.config(state='normal')
            # Clear the existing text
            text_widget.delete("1.0", END)

            # Check if this is the overview topic and if the logo exists ---
            if topic == "General overview" and manual_window.logo_image:
                # Insert the image at the beginning of the text widget
                text_widget.image_create(END, image=manual_window.logo_image)
                # Add a couple of newlines for spacing after the image
                text_widget.insert(END, "\n\n")

            # Insert the main text content
            text_widget.insert(END, content)
            # Disable the text widget to make it read-only
            text_widget.config(state='disabled')

        # --- Create Buttons Dynamically ---
        for topic in aboutprogram.keys():
            # The lambda function with a default argument is crucial here to capture the correct 'topic' for each button
            button = Button(left_frame, text=topic, command=lambda t=topic: update_text(t))
            button.pack(fill='x', pady=2)

        # --- Initial State ---
        # Display the 'General overview' topic by default when the window opens
        update_text("General overview")

        manual_window.lift()
        manual_window.focus_force()
    def _plot_calc_data_action():
        global calc_peak_artists, calc_filename, ax1, fig # Access required globals

        if not calc_filename:
            messagebox.showwarning("No File", "No calculated linelist file has been selected.", parent=dialog)
            return
        if not ax1 or not fig or not fig.canvas:
            messagebox.showerror("Plot Error", "Main plot (ax1) is not available.", parent=dialog)
            return

        print("Plotting calculated linelist...")
        try:
            original_xlim = ax1.get_xlim()
            original_ylim = ax1.get_ylim()
            limits_stored = True
        except Exception as e:
            print(f"Warning: Could not store axes limits before plotting calc data: {e}")
            limits_stored = False

        if calc_peak_artists:
            # print(f"Removing {len(calc_peak_artists)} previous calculated peak artists.") # Optional debug
            for artist in calc_peak_artists:
                try: artist.remove()
                except Exception: pass
            calc_peak_artists.clear()

        if limits_stored:
            try:
                ax1.set_xlim(original_xlim)
                ax1.set_ylim(original_ylim)
            except Exception as e:
                print(f"Warning: Could not restore axes limits before plotting calc data: {e}")

        # Plot new peaks using current settings and store the artists
        calc_peak_artists = plot_calculated_peaks(calc_filename, ax1) # Returns artists

        # Redraw the canvas
        try:
            fig.canvas.draw_idle()
            print("Calculated linelist plot updated.")
        except Exception as e:
            print(f"Error redrawing canvas after plotting calc linelist: {e}")
            messagebox.showerror("Plot Error", f"Error updating plot display:\n{e}", parent=dialog)
    def _plot_peaklist_action():
        global ax1, fig # Access required globals

        # Check if plot is available
        if not ax1 or not fig or not fig.canvas:
            messagebox.showerror("Plot Error", "Main plot (ax1) is not available.", parent=dialog)
            return

        # Call load_existing_peaks - it handles clearing old peaks/annotations
        # and plotting based on current view limits
        load_existing_peaks(peaklist_filename)

        # Redraw the canvas
        try:
            fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error redrawing canvas after plotting Peaklist: {e}")
            messagebox.showerror("Plot Error", f"Error updating plot display:\n{e}", parent=dialog)
    def _toggle_update_on_move():
        global UPDATE_PLOT_ON_MOVE, update_on_move_var
        if update_on_move_var: # Check if the Tkinter variable exists
            UPDATE_PLOT_ON_MOVE = update_on_move_var.get()
            print(f"Upload data set to: {UPDATE_PLOT_ON_MOVE}")

    Button(general_action_button_frame, text="Spectrum Fitting", command=show_fitting_params_dialog).pack(side=LEFT, padx=5)
    Button(general_action_button_frame, text="About program", command=show_manual_window).pack(side=LEFT, padx=5)
    plot_control_frame = Frame(general_action_button_frame)
    plot_control_frame.pack(side=LEFT, padx=(10, 5))
    update_on_move_var = BooleanVar(master=plot_control_frame, value=UPDATE_PLOT_ON_MOVE)
    Checkbutton(plot_control_frame, text="Upload data", variable=update_on_move_var, command=_toggle_update_on_move).pack(side=TOP, pady=5)
    
    # ===================================================================
    # --- CONTOUR SETTINGS PANE ---
    # ===================================================================
    contour_frame.columnconfigure(1, weight=1)
    
    def _update_fwhm_limits(*args):
        selected_type = contour_type_var.get()
        try:
            if selected_type == 'Voigt':
                # Update Gaussian variation limits
                fwhm_g_val = float(fwhm_g_entry.get())
                var_g_percent = float(fwhm_g_var_entry.get())
                if fwhm_g_val > 0 and 0 <= var_g_percent <= 100:
                    lower_g = fwhm_g_val * (1 - var_g_percent / 100)
                    upper_g = fwhm_g_val * (1 + var_g_percent / 100)
                    lower_limit_g_val.config(text=f"{lower_g:.4g}")
                    upper_limit_g_val.config(text=f"{upper_g:.4g}")
                else: raise ValueError("Invalid G range")

                # Update Lorentzian variation limits
                fwhm_l_val = float(fwhm_l_entry.get())
                var_l_percent = float(fwhm_l_var_entry.get())
                if fwhm_l_val > 0 and 0 <= var_l_percent <= 100:
                    lower_l = fwhm_l_val * (1 - var_l_percent / 100)
                    upper_l = fwhm_l_val * (1 + var_l_percent / 100)
                    lower_limit_l_val.config(text=f"{lower_l:.4g}")
                    upper_limit_l_val.config(text=f"{upper_l:.4g}")
                else: raise ValueError("Invalid L range")

            else: # Gaussian or Lorentzian
                fwhm_val = float(fwhm_entry.get())
                var_percent = float(fwhm_var_entry.get())
                if fwhm_val > 0 and 0 <= var_percent <= 100:
                    lower = fwhm_val * (1 - var_percent / 100)
                    upper = fwhm_val * (1 + var_percent / 100)
                    lower_limit_val.config(text=f"{lower:.4g}")
                    upper_limit_val.config(text=f"{upper:.4g}")
                else: raise ValueError("Invalid range")

        except (ValueError, tk.TclError):
            lower_limit_val.config(text="---"); upper_limit_val.config(text="---")
            lower_limit_g_val.config(text="---"); upper_limit_g_val.config(text="---")
            lower_limit_l_val.config(text="---"); upper_limit_l_val.config(text="---")

    # --- This function handles visibility of FWHM entries ---
    def _on_contour_type_change(*args):
        selected_type = contour_type_var.get()
        params = contour_settings['contours'].get(selected_type, {})
        
        # Hide all conditional frames first
        fwhm_frame.grid_forget()
        fwhm_var_frame.grid_forget()
        limits_frame.grid_forget()
        fwhm_g_frame.grid_forget()
        fwhm_l_frame.grid_forget()
        fwhm_g_var_frame.grid_forget()
        limits_g_frame.grid_forget()
        fwhm_l_var_frame.grid_forget()
        limits_l_frame.grid_forget()
        
        if selected_type == 'Voigt':
            # Show Voigt-specific frames and populate them
            fwhm_g_frame.grid(row=1, column=0, columnspan=2, sticky='ew', padx=10, pady=(5,0))
            fwhm_l_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=10, pady=(0,5))
            fwhm_g_var_frame.grid(row=3, column=0, columnspan=2, sticky='ew', padx=10, pady=(5,0))
            limits_g_frame.grid(row=4, column=0, columnspan=2, sticky='ew', padx=10, pady=(0,5))
            fwhm_l_var_frame.grid(row=5, column=0, columnspan=2, sticky='ew', padx=10, pady=(5,0))
            limits_l_frame.grid(row=6, column=0, columnspan=2, sticky='ew', padx=10, pady=(0,5))
            
            fwhm_g_entry.delete(0, END); fwhm_g_entry.insert(0, str(params.get('fwhm_g', DEFAULT_FWHM_G)))
            fwhm_l_entry.delete(0, END); fwhm_l_entry.insert(0, str(params.get('fwhm_l', DEFAULT_FWHM_L)))
            fwhm_g_var_entry.delete(0, END); fwhm_g_var_entry.insert(0, str(params.get('fwhm_g_var', DEFAULT_FWHM_G_VAR)))
            fwhm_l_var_entry.delete(0, END); fwhm_l_var_entry.insert(0, str(params.get('fwhm_l_var', DEFAULT_FWHM_L_VAR)))
        else:
            # Show the generic frames for Gaussian/Lorentzian and populate them
            fwhm_frame.grid(row=1, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
            fwhm_var_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
            limits_frame.grid(row=3, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
            
            fwhm_entry.delete(0, END); fwhm_entry.insert(0, str(params.get('fwhm', DEFAULT_FWHM)))
            fwhm_var_entry.delete(0, END); fwhm_var_entry.insert(0, str(params.get('fwhm_var', DEFAULT_FWHM_VAR)))

        # Center deviation is common to all
        center_dev_entry.delete(0, END); center_dev_entry.insert(0, str(params.get('center_dev', DEFAULT_CENTER_DEV)))
        # Amplitude variation is common to all
        amp_var_entry.delete(0, END)
        amp_var_entry.insert(0, str(params.get('amplitude_var', DEFAULT_AMPLITUDE_VAR)))
        # Populate the new min_peak_dist entry
        min_dist_entry.delete(0, END)
        min_dist_entry.insert(0, str(params.get('min_peak_dist', DEFAULT_MIN_PEAK_DIST)))
        # Populate the sim_window_entry
        sim_window_entry.delete(0, END)
        sim_window_entry.insert(0, str(params.get('sim_window_fwhm', 5.0)))

        _update_fwhm_limits()

    # --- Contour UI Widgets ---
    type_frame = Frame(contour_frame)
    type_frame.grid(row=0, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
    Label(type_frame, text="Contour type:", width=20, anchor='w').pack(side=LEFT)
    contour_type_var = tk.StringVar(dialog, value=contour_settings.get('selected_contour', 'Gaussian'))
    OptionMenu(type_frame, contour_type_var, "Gaussian", "Lorentzian", "Voigt", command=_on_contour_type_change).pack(side=LEFT, fill='x', expand=True)

    # -- Widgets for Gaussian/Lorentzian --
    fwhm_frame = Frame(contour_frame)
    Label(fwhm_frame, text="Default FWHM", width=20, anchor='w').pack(side=LEFT)
    fwhm_entry = Entry(fwhm_frame)
    fwhm_entry.pack(side=LEFT, fill='x', expand=True)
    fwhm_entry.bind("<KeyRelease>", _update_fwhm_limits)

    fwhm_var_frame = Frame(contour_frame)
    Label(fwhm_var_frame, text="FWHM Variation (%)", width=20, anchor='w').pack(side=LEFT)
    fwhm_var_entry = Entry(fwhm_var_frame)
    fwhm_var_entry.pack(side=LEFT, fill='x', expand=True)
    fwhm_var_entry.bind("<KeyRelease>", _update_fwhm_limits)

    limits_frame = Frame(contour_frame)
    Label(limits_frame, text="FWHM variation limits", width=20, anchor='w').pack(side=LEFT)
    limit_widgets_frame = Frame(limits_frame)
    limit_widgets_frame.pack(side=LEFT, fill='x', expand=True)
    lower_limit_val = Label(limit_widgets_frame, text="", width=12, relief="sunken", anchor='w')
    lower_limit_val.pack(side=LEFT, fill='x', expand=True)
    Label(limit_widgets_frame, text=" to ").pack(side=LEFT, padx=5)
    upper_limit_val = Label(limit_widgets_frame, text="", width=12, relief="sunken", anchor='w')
    upper_limit_val.pack(side=LEFT, fill='x', expand=True)

    # -- Widgets for Voigt --
    fwhm_g_frame = Frame(contour_frame)
    Label(fwhm_g_frame, text="Default FWHM (Gaussian)", width=20, anchor='w').pack(side=LEFT)
    fwhm_g_entry = Entry(fwhm_g_frame)
    fwhm_g_entry.pack(side=LEFT, fill='x', expand=True)
    fwhm_g_entry.bind("<KeyRelease>", _update_fwhm_limits)
    
    fwhm_l_frame = Frame(contour_frame)
    Label(fwhm_l_frame, text="Default FWHM (Lorentzian)", width=20, anchor='w').pack(side=LEFT)
    fwhm_l_entry = Entry(fwhm_l_frame)
    fwhm_l_entry.pack(side=LEFT, fill='x', expand=True)
    fwhm_l_entry.bind("<KeyRelease>", _update_fwhm_limits)

    fwhm_g_var_frame = Frame(contour_frame)
    Label(fwhm_g_var_frame, text="FWHM Gaussian Variation (%)", font=('Arial', 9, 'bold'), width=25, anchor='w').pack(side=LEFT)
    fwhm_g_var_entry = Entry(fwhm_g_var_frame)
    fwhm_g_var_entry.pack(side=LEFT, fill='x', expand=True)
    fwhm_g_var_entry.bind("<KeyRelease>", _update_fwhm_limits)

    limits_g_frame = Frame(contour_frame)
    Label(limits_g_frame, text="FWHM variation limits", width=25, anchor='w').pack(side=LEFT)
    limit_g_widgets_frame = Frame(limits_g_frame)
    limit_g_widgets_frame.pack(side=LEFT, fill='x', expand=True)
    lower_limit_g_val = Label(limit_g_widgets_frame, text="", width=12, relief="sunken", anchor='w')
    lower_limit_g_val.pack(side=LEFT, fill='x', expand=True)
    Label(limit_g_widgets_frame, text=" to ").pack(side=LEFT, padx=5)
    upper_limit_g_val = Label(limit_g_widgets_frame, text="", width=12, relief="sunken", anchor='w')
    upper_limit_g_val.pack(side=LEFT, fill='x', expand=True)

    fwhm_l_var_frame = Frame(contour_frame)
    Label(fwhm_l_var_frame, text="FWHM Lorentzian Variation (%)", font=('Arial', 9, 'bold'), width=25, anchor='w').pack(side=LEFT)
    fwhm_l_var_entry = Entry(fwhm_l_var_frame)
    fwhm_l_var_entry.pack(side=LEFT, fill='x', expand=True)
    fwhm_l_var_entry.bind("<KeyRelease>", _update_fwhm_limits)
    
    limits_l_frame = Frame(contour_frame)
    Label(limits_l_frame, text="FWHM variation limits", width=25, anchor='w').pack(side=LEFT)
    limit_l_widgets_frame = Frame(limits_l_frame)
    limit_l_widgets_frame.pack(side=LEFT, fill='x', expand=True)
    lower_limit_l_val = Label(limit_l_widgets_frame, text="", width=12, relief="sunken", anchor='w')
    lower_limit_l_val.pack(side=LEFT, fill='x', expand=True)
    Label(limit_l_widgets_frame, text=" to ").pack(side=LEFT, padx=5)
    upper_limit_l_val = Label(limit_l_widgets_frame, text="", width=12, relief="sunken", anchor='w')
    upper_limit_l_val.pack(side=LEFT, fill='x', expand=True)

    # -- Common Widget (Center Deviation) --
    center_dev_frame = Frame(contour_frame)
    center_dev_frame.grid(row=8, column=0, columnspan=2, sticky='ew', padx=10, pady=(10, 5)) 
    Label(center_dev_frame, text="Center deviation", width=20, anchor='w').pack(side=LEFT)
    center_dev_entry = Entry(center_dev_frame)
    center_dev_entry.pack(side=LEFT, fill='x', expand=True)

    # -- Common Widget (Minimal Peak Distance) --
    min_dist_frame = Frame(contour_frame)
    min_dist_frame.grid(row=9, column=0, columnspan=2, sticky='ew', padx=10, pady=5) # Placed at row 9
    Label(min_dist_frame, text="Minimal Peak Distance", width=20, anchor='w').pack(side=LEFT)
    min_dist_entry = Entry(min_dist_frame)
    min_dist_entry.pack(side=LEFT, fill='x', expand=True)

    amp_var_frame = Frame(contour_frame)
    amp_var_frame.grid(row=7, column=0, columnspan=2, sticky='ew', padx=10, pady=5)
    Label(amp_var_frame, text="Amplitude Variation (%)", width=20, anchor='w').pack(side=LEFT)
    amp_var_entry = Entry(amp_var_frame)
    amp_var_entry.pack(side=LEFT, fill='x', expand=True)
    
    # Simulation Window Entry
    sim_window_frame = Frame(contour_frame)
    sim_window_frame.grid(row=10, column=0, columnspan=2, sticky='ew', padx=10, pady=(5,5))
    Label(sim_window_frame, text="Simulation Window (in FWHM)", width=25, anchor='w').pack(side=LEFT)
    sim_window_entry = Entry(sim_window_frame)
    sim_window_entry.pack(side=LEFT, fill='x', expand=True)

    # Multi-Start Attempts Entry ---
    num_starts_frame = Frame(contour_frame)
    num_starts_frame.grid(row=11, column=0, columnspan=2, sticky='ew', padx=10, pady=(5,5))
    Label(num_starts_frame, text="Multi-Start Attempts", width=25, anchor='w').pack(side=LEFT)
    num_starts_entry = Entry(num_starts_frame)
    num_starts_entry.pack(side=LEFT, fill='x', expand=True)
    # Populate with the current value from settings
    num_starts_entry.insert(0, str(current_settings.get('num_starts', DEFAULT_NUM_STARTS)))

    # Initialize the UI
    _on_contour_type_change()

    # ===================================================================
    # --- PLOT SETTINGS PANE ---
    # ===================================================================
    # (Plot pane UI is unchanged)
    dialog.current_graph_num = tk.IntVar(master=dialog, value=1)
    dialog.current_plot_num = tk.IntVar(master=dialog, value=1)
    dialog.plot_type_var = tk.StringVar(master=dialog, value="None")
    dialog.plot_filepath_var = tk.StringVar(master=dialog, value="")
    plot_nav_frame = Frame(plot_frame); plot_nav_frame.pack(fill='x', padx=10, pady=10)
    plot_details_frame = Frame(plot_frame, bd=2, relief="groove"); plot_details_frame.pack(fill='both', expand=True, padx=10, pady=5)
    def _get_current_plot_data():
        g_num, p_num = dialog.current_graph_num.get(), dialog.current_plot_num.get()
        if g_num not in plot_settings: plot_settings[g_num] = {}
        if p_num not in plot_settings[g_num]: plot_settings[g_num][p_num] = {'type': 'None', 'file': ''}
        return plot_settings[g_num][p_num]
    def _save_current_plot_settings():
        plot_data = _get_current_plot_data()
        plot_data['type'], plot_data['file'] = dialog.plot_type_var.get(), dialog.plot_filepath_var.get()
    def _load_and_display_plot_settings():
        plot_data = _get_current_plot_data()
        dialog.plot_type_var.set(plot_data.get('type', 'None'))
        dialog.filepath_label.config(text=plot_data.get('file', ''))
        _on_plot_type_change()
    def _update_plot_sliders(delta, which):
        _save_current_plot_settings()
        var, label = (dialog.current_graph_num, dialog.graph_num_label) if which == 'graph' else (dialog.current_plot_num, dialog.plot_num_label)
        new_val = var.get() + delta
        if new_val >= 1: var.set(new_val); label.config(text=str(new_val))
        _load_and_display_plot_settings()
    def _on_plot_type_change(*args):
        if dialog.plot_type_var.get() == "2-D graph": plot_file_frame.pack(fill='x', padx=10, pady=5)
        else: plot_file_frame.pack_forget(); dialog.filepath_label.config(text="")
    def _select_plot_file():
        filepath = filedialog.askopenfilename(parent=dialog)
        if filepath: dialog.filepath_label.config(text=filepath); _save_current_plot_settings()
    Label(plot_nav_frame, text="Graph #:").pack(side=tk.LEFT); Button(plot_nav_frame, text="<", command=lambda: _update_plot_sliders(-1, 'graph')).pack(side=tk.LEFT)
    dialog.graph_num_label = Label(plot_nav_frame, text="1", width=4, relief='sunken'); dialog.graph_num_label.pack(side=tk.LEFT)
    Button(plot_nav_frame, text=">", command=lambda: _update_plot_sliders(1, 'graph')).pack(side=tk.LEFT)
    Label(plot_nav_frame, text="Plot #:").pack(side=tk.LEFT, padx=(20, 0)); Button(plot_nav_frame, text="<", command=lambda: _update_plot_sliders(-1, 'plot')).pack(side=tk.LEFT)
    dialog.plot_num_label = Label(plot_nav_frame, text="1", width=4, relief='sunken'); dialog.plot_num_label.pack(side=tk.LEFT)
    Button(plot_nav_frame, text=">", command=lambda: _update_plot_sliders(1, 'plot')).pack(side=tk.LEFT)
    type_frame = Frame(plot_details_frame); type_frame.pack(fill='x', padx=10, pady=10)
    Label(type_frame, text="Plot Type:", width=12, anchor='w').pack(side=tk.LEFT)
    OptionMenu(type_frame, dialog.plot_type_var, "None", "2-D graph", command=_on_plot_type_change).pack(side=tk.LEFT, fill='x', expand=True)
    plot_file_frame = Frame(plot_details_frame)
    Label(plot_file_frame, text="File Path:", width=12, anchor='w').pack(side=tk.LEFT)
    Button(plot_file_frame, text="Open File...", command=_select_plot_file).pack(side=tk.RIGHT, padx=5)
    dialog.filepath_label = Label(plot_file_frame, text="", relief='sunken', anchor='w'); dialog.filepath_label.pack(side=tk.LEFT, fill='x', expand=True)
    _load_and_display_plot_settings()
    
    # ===================================================================
    # --- DIALOG-WIDE BUTTONS AND LOGIC ---
    # ===================================================================
    def save_all_changes():
        _save_current_plot_settings()
        nonlocal dialog
        global current_settings, DEFAULT_FWHM, FWHM_VAR, EXT_SCALE, CALC_SCALE
        global INTENSITY_THRESH, CENTER_DEV, REG_PARAM, ALPHA, STATE_INDEX
        global MIN_AMP, calc_peak_artists, contour_settings, peaklist_filename
        global DEFAULT_FWHM_G, DEFAULT_FWHM_L, DEFAULT_CONTOUR
        global DEFAULT_FWHM_G_VAR, DEFAULT_FWHM_L_VAR, DEFAULT_AMPLITUDE_VAR 

        new_values = {}
        errors = []
        requires_redraw = False

        # --- Save Contour Settings First ---
        try:
            selected_type = contour_type_var.get()
            contour_settings['selected_contour'] = selected_type
            if selected_type not in contour_settings['contours']:
                contour_settings['contours'][selected_type] = {}
            
            if selected_type == 'Voigt':
                c_fwhm_g = float(fwhm_g_entry.get())
                c_fwhm_l = float(fwhm_l_entry.get())
                c_fwhm_g_var = float(fwhm_g_var_entry.get())
                c_fwhm_l_var = float(fwhm_l_var_entry.get())
                
                if not (c_fwhm_g > 0.0 and c_fwhm_l > 0.0): errors.append("Voigt FWHM components must be positive.")
                if not (0.0 <= c_fwhm_g_var <= 100.0): errors.append("Gaussian Variation must be 0-100.")
                if not (0.0 <= c_fwhm_l_var <= 100.0): errors.append("Lorentzian Variation must be 0-100.")
                
                if not errors:
                    contour_settings['contours'][selected_type]['fwhm_g'] = c_fwhm_g
                    contour_settings['contours'][selected_type]['fwhm_l'] = c_fwhm_l
                    contour_settings['contours'][selected_type]['fwhm_g_var'] = c_fwhm_g_var
                    contour_settings['contours'][selected_type]['fwhm_l_var'] = c_fwhm_l_var
            else: # Gaussian or Lorentzian
                c_fwhm = float(fwhm_entry.get())
                c_fwhm_var = float(fwhm_var_entry.get())
                if not (c_fwhm > 0.0): errors.append("Default FWHM must be positive.")
                if not (0.0 <= c_fwhm_var <= 100.0): errors.append("FWHM Variation must be 0-100.")
                
                if not errors:
                    contour_settings['contours'][selected_type]['fwhm'] = c_fwhm
                    contour_settings['contours'][selected_type]['fwhm_var'] = c_fwhm_var

            # Common parameters
            c_center_dev = float(center_dev_entry.get())
            if not (c_center_dev >= 0.0): errors.append("Center deviation must be non-negative.")

            # Get and validate the new amplitude variation parameter
            c_amp_var = float(amp_var_entry.get())
            if not (c_amp_var >= 0.0): errors.append("Amplitude Variation must be a non-negative number.")
            
            # Get and validate the simulation window parameter
            c_sim_window = float(sim_window_entry.get())
            if not (c_sim_window > 0.0): errors.append("Simulation Window must be a positive number.")

            # Get and validate the new minimal peak distance parameter
            c_min_dist = float(min_dist_entry.get())
            if not (c_min_dist >= 0.0): errors.append("Minimal Peak Distance must be non-negative.")

            if not errors:
                contour_settings['contours'][selected_type]['center_dev'] = c_center_dev
                contour_settings['contours'][selected_type]['amplitude_var'] = c_amp_var
                contour_settings['contours'][selected_type]['sim_window_fwhm'] = c_sim_window
                contour_settings['contours'][selected_type]['min_peak_dist'] = c_min_dist
                new_values.update(contour_settings['contours'][selected_type])

        except ValueError:
            errors.append("Contour parameters must be valid numbers.")

        # Save Multi-Start Attempts from Contour Pane ---
        try:
            c_num_starts = int(num_starts_entry.get())
            if not (c_num_starts >= 1):
                errors.append("Multi-Start Attempts must be an integer >= 1.")
            else:
                new_values['num_starts'] = c_num_starts
        except ValueError:
            errors.append("Multi-Start Attempts must be a valid integer.")

        # --- Save General Settings ---
        for config in settings_config:
            key = config['key']
            widget_info = general_settings_widgets[key]
            if config['type'] == 'int_spinner':
                value = widget_info['current_value']
                if not config['validation'](value): errors.append(f"Invalid value for {config['label']}: {value}")
                else: new_values[key] = value
            else:
                value_str = widget_info['entry'].get().strip()
                try:
                    value = config['type'](value_str)
                    if not config['validation'](value): errors.append(f"Validation failed for {config['label']}")
                    else: new_values[key] = value
                except ValueError:
                    errors.append(f"Invalid format for {config['label']}")

        # Save Controls Settings ---
        for key, var in controls_widgets.items():
            new_values[key] = var.get() # .get() on a BooleanVar returns True/False

        # Get and validate the new peaklist file path
        new_peaklist_path = dialog.peaklist_var.get()
        if not new_peaklist_path:
            errors.append("Peaklist database file path cannot be empty.")
        else:
            new_values['peaklist_file'] = new_peaklist_path

        if errors:
            messagebox.showerror("Settings Error", "Please fix errors:\n- " + "\n- ".join(errors), parent=dialog)
            return

        # --- Update current_settings and check for redraw ---
        old_peaklist_path = peaklist_filename
        all_configs = settings_config + [{'key': k, 'redraw_trigger': False} for k in ['fwhm', 'fwhm_var', 'center_dev']]
        for key, value in new_values.items():
            if current_settings.get(key) != value:
                config_item = next((c for c in all_configs if c['key'] == key), None)
                if config_item and config_item.get('redraw_trigger'):
                    requires_redraw = True
            current_settings[key] = value

        # Update globals from the final current_settings
        peaklist_filename = current_settings['peaklist_file']
        DEFAULT_FWHM = current_settings.get('fwhm', 0.1)
        FWHM_VAR = current_settings.get('fwhm_var', 0.0)
        DEFAULT_FWHM_G = current_settings.get('fwhm_g', 0.05)
        DEFAULT_FWHM_L = current_settings.get('fwhm_l', 0.05)
        DEFAULT_FWHM_G_VAR = current_settings.get('fwhm_g_var', 0.0)
        DEFAULT_FWHM_L_VAR = current_settings.get('fwhm_l_var', 0.0)
        DEFAULT_AMPLITUDE_VAR = current_settings.get('amplitude_var', DEFAULT_AMPLITUDE_VAR)
        EXT_SCALE = current_settings['ext_scale']
        CALC_SCALE = current_settings['calc_scale']
        INTENSITY_THRESH = current_settings['intensity_thresh']
        CENTER_DEV = current_settings['center_dev']
        REG_PARAM = current_settings['reg_param']
        ALPHA = current_settings['alpha']
        STATE_INDEX = current_settings['state_index']
        DEFAULT_CONTOUR = contour_settings.get('selected_contour', 'Gaussian')
        MIN_PEAK_DIST = current_settings.get('min_peak_dist', DEFAULT_MIN_PEAK_DIST)
        MARKER_DIAMETER = current_settings.get('marker_diameter', DEFAULT_MARKER_DIAMETER)

        save_settings_to_file(current_settings, plot_settings)
        dialog.destroy()
        globals()['settings_window_open'] = False
        print("Settings updated.")

        # Clear old peaks if the file changed, but DO NOT load new ones
        if old_peaklist_path != peaklist_filename:
            print(f"Peaklist file changed from '{old_peaklist_path}' to '{peaklist_filename}'.")
            print("Clearing old peak data. Use 'Plot Peaklist' to view data from the new file.")
            clear_all_peak_data() # This handles removal from plot and memory
        elif requires_redraw:
            # (original redraw logic is unchanged)
            print("Redrawing plot due to changed settings while preserving view...")
            try:
                original_xlim, original_ylim = ax1.get_xlim(), ax1.get_ylim()
                ax1.clear(); calc_peak_artists.clear(); peaklist_annotation_artists.clear()
                ax1.plot(x, y, 'b-', label='Raw Data')
                load_existing_peaks(peaklist_filename)
                ax1.set_xlim(original_xlim); ax1.set_ylim(original_ylim)
                calc_peak_artists = plot_calculated_peaks(calc_filename, ax1)
                fig.canvas.draw_idle()
                print("Plot redraw complete.")
            except Exception as e:
                print(f"Error during plot redraw: {e}")

    def close_dialog_and_flag():
        dialog.destroy()
        globals()['settings_window_open'] = False

    Button(bottom_button_frame, text="Cancel", command=close_dialog_and_flag).pack(side=RIGHT, padx=5)
    Button(bottom_button_frame, text="Save", command=save_all_changes).pack(side=RIGHT, padx=5)
    
    dialog.protocol("WM_DELETE_WINDOW", close_dialog_and_flag)
    switch_group("General") 
    dialog.lift()
    dialog.attributes('-topmost', True)
    dialog.after_idle(dialog.attributes, '-topmost', False)
    dialog.focus_force()

def show_fitting_params_dialog():
    """Dialog for spectrum fitting parameters and execution."""
   # This dialog directly modifies the global variables for the current session
    # and the settings dictionaries for persistence.
    global fitting_params_window_open, current_settings, contour_settings
    global DEFAULT_FWHM, FWHM_VAR, CENTER_DEV, REG_PARAM, ALPHA, MIN_AMP, NUM_STARTS

    if fitting_params_window_open:
        return
    fitting_params_window_open = True

    # --- Fitting Parameters Configuration ---
    # Similar structure to settings_config, but focuses on fitting params
    fitting_params_config = [
        {'key': 'fwhm', 'label': 'Default FWHM:', 'type': float,
         'validation': lambda v: v > 0.0, 'tooltip': '# Initial FWHM guess (> 0.0)'},
        {'key': 'fwhm_var', 'label': 'FWHM Variation (%):', 'type': float,
         'validation': lambda v: 0.0 <= v <= 100.0, 'tooltip': '# Allowed FWHM variation (0-100 %)'},
        {'key': 'center_dev', 'label': 'Center Deviation:', 'type': float,
         'validation': lambda v: v >= 0.0, 'tooltip': '# Max allowed shift in X (>= 0.0)'},
        {'key': 'reg_param', 'label': 'Linear Regularization:', 'type': float,
         'validation': lambda v: v >= 0.0, 'tooltip': '# Regularization penalty (>=0.0)'},
        {'key': 'alpha', 'label': 'Alpha Weight (0-1):', 'type': float,
         'validation': lambda v: 0.0 <= v <= 1.0, 'tooltip': '# Residual weighting (0=contour only)'},
        {'key': 'min_amp', 'label': 'Minimum Amplitude:', 'type': float,
         'validation': lambda v: v >= 0.0, 'tooltip': '# Minimum amplitude for peak inclusion (>= 0.0)'},
    ]

    # Get parent window
    fig_manager = plt.get_current_fig_manager()
    root = fig_manager.window if fig_manager else Tk()

    dialog = Toplevel(root)
    dialog.title("Spectrum Fitting Parameters")
    position_dialog_near_cursor(dialog, width=550, height=300)

    fitting_widgets = {} # Store Entry widgets

    # --- Create UI Elements Dynamically ---
    for config in fitting_params_config:
        key = config['key']
        frame = Frame(dialog)
        frame.pack(padx=10, pady=5, fill='x')
        Label(frame, text=config['label'], width=20, anchor='w').pack(side='left')

        entry = Entry(frame)
        current_val = current_settings.get(key) # Get value from central dict
        if current_val is not None:
            entry.insert(0, str(current_val))
        entry.pack(side='left', fill='x', expand=True)
        fitting_widgets[key] = entry # Store widget

        Label(frame, text=config['tooltip'], fg='gray', font=('Arial', 8)).pack(side='left', padx=5)

    # --- Save Function ---
    def save_fitting_params():
        nonlocal dialog
        # Access all required globals to ensure they are updated
        global current_settings, contour_settings
        global DEFAULT_FWHM, FWHM_VAR, CENTER_DEV, REG_PARAM, ALPHA, MIN_AMP, NUM_STARTS

        new_values = {}
        errors = []

        for config in fitting_params_config:
            key = config['key']
            entry_widget = fitting_widgets[key]
            value_str = entry_widget.get().strip()
            try:
                value = config['type'](value_str) 
                if not config['validation'](value):
                    errors.append(f"Validation failed for {config['label']}. Input: '{value_str}'")
                else:
                    new_values[key] = value
            except ValueError:
                errors.append(f"Invalid format for {config['label']}. Expected {config['type'].__name__}. Input: '{value_str}'")

        if errors:
            messagebox.showerror("Input Error", "Please fix the following errors:\n- " + "\n- ".join(errors), parent=dialog)
            return

        print("Updating Fitting Parameters:")
        selected_contour_type = contour_settings.get('selected_contour', 'Gaussian')

        for key, value in new_values.items():
            if current_settings.get(key) != value:
                print(f"  {key}: {current_settings.get(key)} -> {value}")
                
                # 1. Update the central settings dictionary for the current session
                current_settings[key] = value
                
                # 2. Update the persistent contour_settings data structure
                # This ensures the change is correctly saved to settings.txt
                if key in ['fwhm', 'fwhm_var', 'center_dev']:
                    if selected_contour_type in contour_settings['contours']:
                        contour_settings['contours'][selected_contour_type][key] = value
                    else:
                        print(f"Warning: Selected contour '{selected_contour_type}' not found in settings. Cannot save {key}.")

                # 3. Update the corresponding global variable for immediate use by fit_peaks
                global_var_name = key.upper()
                if key == 'fwhm': global_var_name = 'DEFAULT_FWHM'
                if key == 'fwhm_var': global_var_name = 'FWHM_VAR'
                
                if global_var_name in globals():
                     globals()[global_var_name] = value
                else:
                     print(f"Warning: Global variable {global_var_name} not found for setting key {key}")

        save_settings_to_file(current_settings, plot_settings)
        
        dialog.destroy()
        globals()['fitting_params_window_open'] = False
        print("Fitting parameters updated and settings saved.")

    button_frame = Frame(dialog)
    button_frame.pack(pady=15, fill='x', padx=10)

    def close_fitting_dialog_and_flag():
        dialog.destroy()
        globals()['fitting_params_window_open'] = False

    Button(button_frame, text="Save", command=save_fitting_params).pack(side=LEFT, padx=10)
    Button(button_frame, text="Cancel", command=close_fitting_dialog_and_flag).pack(side=RIGHT, padx=10)

    dialog.protocol("WM_DELETE_WINDOW", close_fitting_dialog_and_flag)
    dialog.lift()
    dialog.attributes('-topmost', True)
    dialog.after_idle(dialog.attributes, '-topmost', False)
    dialog.focus_force()


def load_settings():
    """Load settings, now with support for structured contour and plot settings."""
    global plot_settings, contour_settings
    plot_settings.clear()

    DEFAULT_CONTOUR_SETTINGS = {
        'selected_contour': 'Gaussian',
        'contours': {
            'Gaussian': {
                'fwhm': DEFAULT_FWHM,
                'fwhm_var': DEFAULT_FWHM_VAR,
                'center_dev': DEFAULT_CENTER_DEV,
                'amplitude_var': DEFAULT_AMPLITUDE_VAR,
                'sim_window_fwhm': 5.0,
                'min_peak_dist': DEFAULT_MIN_PEAK_DIST,
            },
            'Lorentzian': {
                'fwhm': DEFAULT_FWHM,
                'fwhm_var': DEFAULT_FWHM_VAR,
                'center_dev': DEFAULT_CENTER_DEV,
                'amplitude_var': DEFAULT_AMPLITUDE_VAR,
                'sim_window_fwhm': 5.0,
                'min_peak_dist': DEFAULT_MIN_PEAK_DIST,
            },
            'Voigt': {
                'fwhm_g': DEFAULT_FWHM_G,
                'fwhm_l': DEFAULT_FWHM_L,
                'fwhm_g_var': DEFAULT_FWHM_G_VAR, 
                'fwhm_l_var': DEFAULT_FWHM_L_VAR, 
                'center_dev': DEFAULT_CENTER_DEV,
                'amplitude_var': DEFAULT_AMPLITUDE_VAR,
                'sim_window_fwhm': 5.0,
                'min_peak_dist': DEFAULT_MIN_PEAK_DIST,
            }
        }
    }
    # Initialize with a deep copy of the defaults
    import copy
    contour_settings = copy.deepcopy(DEFAULT_CONTOUR_SETTINGS)

    # General settings dictionary (will become current_settings)
    settings = {
        'ext_scale': DEFAULT_EXT_SCALE,
        'calc_scale': DEFAULT_CALC_SCALE,
        'intensity_thresh': DEFAULT_INTENSITY_THRESH,
        'reg_param': DEFAULT_REG_PARAM,
        'alpha': DEFAULT_ALPHA,
        'min_amp': DEFAULT_MIN_AMP,
        'state_index': DEFAULT_STATE_INDEX,
        'num_starts': DEFAULT_NUM_STARTS,
        'peaklist_file': DEFAULT_PEAKLIST_FILE,
        'marker_diameter': DEFAULT_MARKER_DIAMETER,
        'disable_mpl_controls': DEFAULT_DISABLE_MPL_CONTROLS,
    }
    param_map = {
        'EXT_SCALE': ('ext_scale', float),
        'CALC_SCALE': ('calc_scale', float),
        'INTENSITY_THRESH': ('intensity_thresh', float),
        'REG_PARAM': ('reg_param', float),
        'ALPHA': ('alpha', float),
        'MIN_AMP': ('min_amp', float),
        'STATE_INDEX': ('state_index', int),
        'NUM_STARTS': ('num_starts', int),
        'PEAKLIST_FILE': ('peaklist_file', str),
        'MARKER_DIAMETER': ('marker_diameter', float),
        'DISABLE_MPL_CONTROLS': ('disable_mpl_controls', lambda v: v.lower() == 'true'),
    }

    try:
        if os.path.exists(SETTINGS_FILE):
            in_plot_settings_block = False
            in_contour_settings_block = False
            with open(SETTINGS_FILE, 'r') as f:
                current_param_key = None
                for line in f:
                    line = line.strip()
                    if not line: continue

                    # --- Block Management ---
                    if line.startswith('#PLOT_SETTINGS_START'): in_plot_settings_block = True; continue
                    if line.startswith('#PLOT_SETTINGS_END'): in_plot_settings_block = False; continue
                    if line.startswith('#CONTOUR_SETTINGS_START'): in_contour_settings_block = True; continue
                    if line.startswith('#CONTOUR_SETTINGS_END'): in_contour_settings_block = False; continue

                    # --- Contour Settings Parsing ---
                    if in_contour_settings_block:
                        try:
                            parts = line.split(';', 3)
                            if parts[0] == 'selected' and len(parts) >= 2:
                                contour_settings['selected_contour'] = parts[1]
                            elif parts[0] == 'param' and len(parts) >= 4:
                                c_type, c_key, c_val_str = parts[1], parts[2], parts[3]
                                if c_type not in contour_settings['contours']:
                                    contour_settings['contours'][c_type] = {}
                                
                                # Check the default structure for the given contour type first
                                default_val = DEFAULT_CONTOUR_SETTINGS['contours'].get(c_type, {}).get(c_key)
                                # Fallback to Gaussian if the type or key is new
                                if default_val is None:
                                    default_val = DEFAULT_CONTOUR_SETTINGS['contours']['Gaussian'].get(c_key)

                                if default_val is not None:
                                    contour_settings['contours'][c_type][c_key] = type(default_val)(c_val_str)
                        except Exception as e:
                            print(f"Warning: Skipping malformed contour setting line: '{line}'. Error: {e}")
                        continue
                    
                    # --- Plot Settings Parsing ---
                    if in_plot_settings_block:
                        # (This block is unchanged)
                        try:
                            parts = line.split(';', 3)
                            if len(parts) >= 3:
                                g_num, p_num, p_type = int(parts[0]), int(parts[1]), parts[2]
                                p_file = parts[3] if len(parts) > 3 else ''
                                if g_num not in plot_settings: plot_settings[g_num] = {}
                                plot_settings[g_num][p_num] = {'type': p_type, 'file': p_file}
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Skipping malformed plot setting line: '{line}'. Error: {e}")
                        continue

                    # --- General settings parsing (no change) ---
                    if line.startswith('#'):
                        current_param_key = line[1:].strip()
                    elif current_param_key in param_map:
                        dict_key, converter = param_map[current_param_key]
                        try:
                            settings[dict_key] = converter(line)
                        except ValueError as e:
                            print(f"Warning: Invalid value '{line}' for {current_param_key}...")
                        current_param_key = None
        else:
            print("Settings file not found. Creating with default values.")

    except Exception as e:
        print(f"Error loading settings: {str(e)}, using defaults.")
    
    # --- CRITICAL STEP ---
    # Populate the main settings dict with the parameters from the SELECTED contour
    # This ensures backward compatibility with functions that use current_settings
    selected_c_type = contour_settings.get('selected_contour', 'Gaussian')
    if selected_c_type in contour_settings['contours']:
        settings.update(contour_settings['contours'][selected_c_type])
    else: # Fallback if selected contour doesn't exist in the data
        settings.update(contour_settings['contours']['Gaussian'])

    # Save defaults if file doesn't exist or is empty
    if not os.path.exists(SETTINGS_FILE) or os.path.getsize(SETTINGS_FILE) == 0:
         save_settings_to_file(settings, plot_settings) # `settings` now includes the contour params

    return settings

def _update_plots_for_view():
    """
    Helper function to replot calculated linelist and peaklist
    if the UPDATE_PLOT_ON_MOVE flag is True.
    Called after pan/zoom operations.
    """
    global UPDATE_PLOT_ON_MOVE, ax1, fig, calc_filename, calc_peak_artists

    if not UPDATE_PLOT_ON_MOVE:
        return # Do nothing if the flag is not set

    if not ax1 or not fig or not fig.canvas:
        # Should not happen if events are firing, but safety check
        print("Warning (_update_plots_for_view): Plot not available.")
        return

    print("Auto-updating plots for new view...")

    # --- Update Calculated Linelist ---
    if calc_filename: # Only if a file is loaded
        # Remove old artists
        if calc_peak_artists:
            for artist in calc_peak_artists:
                try: artist.remove()
                except Exception: pass
            calc_peak_artists.clear()
        # Replot based on current view (plot_calculated_peaks handles filtering)
        # We assume limits are already set correctly by the event handler
        calc_peak_artists = plot_calculated_peaks(calc_filename, ax1)
    # else: No calculated linelist to update

    # --- Update Peaklist ---
    # load_existing_peaks handles clearing old + plotting based on current view
    load_existing_peaks(peaklist_filename)

    # --- Redraw ---
    try:
        fig.canvas.draw_idle()
        # print("Auto-update complete.") # Optional debug
    except Exception as e:
        print(f"Error during auto-redraw: {e}")

# --- New Helper Function to Save Settings to File ---
# This separates the file writing logic
def save_settings_to_file(settings_dict, plot_settings_dict):
    """Writes the provided settings dictionaries to the SETTINGS_FILE."""
    global contour_settings # We need access to the new global

    # Contour-related keys are now handled by the new block, so they are
    # removed from this general configuration list.
    write_config = [
        ('EXT_SCALE', 'ext_scale', "{:.6f}"),
        ('CALC_SCALE', 'calc_scale', "{:.6f}"),
        ('INTENSITY_THRESH', 'intensity_thresh', "{:.6f}"),
        ('REG_PARAM', 'reg_param', "{:.8f}"),
        ('ALPHA', 'alpha', "{:.6f}"),
        ('MIN_AMP', 'min_amp', "{:.6f}"),
        ('STATE_INDEX', 'state_index', "{}"),
        ('NUM_STARTS', 'num_starts', "{}"),
        ('PEAKLIST_FILE', 'peaklist_file', "{}"),
        ('MARKER_DIAMETER', 'marker_diameter', "{:.1f}"),
        ('DISABLE_MPL_CONTROLS', 'disable_mpl_controls', "{}"),
    ]
    try:
        with open(SETTINGS_FILE, 'w') as f:
            # Contour Settings Writing Logic
            f.write("#CONTOUR_SETTINGS_START\n")
            # Write the selected contour type
            f.write(f"selected;{contour_settings.get('selected_contour', 'Gaussian')}\n")
            # Write the parameters for all known contour types
            for c_type, params in contour_settings.get('contours', {}).items():
                for key, value in params.items():
                    f.write(f"param;{c_type};{key};{value}\n")
            f.write("#CONTOUR_SETTINGS_END\n\n")
            # --- [* END NEW LOGIC *] ---
            
            # --- Write general settings ---
            for file_key, dict_key, fmt in write_config:
                f.write(f"#{file_key}\n")
                value = settings_dict.get(dict_key)
                if value is None: continue
                f.write(fmt.format(value) + "\n")

            # --- Write Plot Settings ---
            # (This block is unchanged)
            f.write("\n#PLOT_SETTINGS_START\n")
            sorted_graphs = sorted(plot_settings_dict.keys())
            for g_num in sorted_graphs:
                sorted_plots = sorted(plot_settings_dict[g_num].keys())
                for p_num in sorted_plots:
                    plot_data = plot_settings_dict[g_num][p_num]
                    if plot_data.get('type', 'None') != 'None' or plot_data.get('file', ''):
                        p_type = plot_data.get('type', 'None')
                        p_file = plot_data.get('file', '')
                        f.write(f"{g_num};{p_num};{p_type};{p_file}\n")
            f.write("#PLOT_SETTINGS_END\n")

        print(f"Settings saved to {SETTINGS_FILE}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to write settings to {SETTINGS_FILE}: {e}")
        messagebox.showerror("File Error", f"Could not save settings to {SETTINGS_FILE}:\n{e}")


# --- Update Global Variable Initialization ---
# Load settings into a dictionary first
current_settings = load_settings()

peaklist_filename = current_settings.get('peaklist_file', DEFAULT_PEAKLIST_FILE)
contour_settings = contour_settings # This line is just to prevent linting errors if contour_settings is not used elsewhere immediately

# Update globals from the loaded dictionary
# (Keep globals for now to minimize changes in the rest of the code,
# but the settings window will primarily interact with current_settings dict)
DEFAULT_FWHM = current_settings.get('fwhm', 0.1)

# For Voigt contours. This now correctly reads the loaded fwhm_g and fwhm_l
# from the settings file, or falls back to the hardcoded defaults.
DEFAULT_FWHM_G = current_settings.get('fwhm_g', DEFAULT_FWHM_G)
DEFAULT_FWHM_L = current_settings.get('fwhm_l', DEFAULT_FWHM_L)

FWHM_VAR = current_settings.get('fwhm_var', 0.0)

# For Voigt contours (Variation %)
DEFAULT_FWHM_G_VAR = current_settings.get('fwhm_g_var', DEFAULT_FWHM_G_VAR)
DEFAULT_FWHM_L_VAR = current_settings.get('fwhm_l_var', DEFAULT_FWHM_L_VAR)

DEFAULT_AMPLITUDE_VAR = current_settings.get('amplitude_var', DEFAULT_AMPLITUDE_VAR)

EXT_SCALE = current_settings.get('ext_scale', 1.0)
CALC_SCALE = current_settings.get('calc_scale', 1.0)
INTENSITY_THRESH = current_settings.get('intensity_thresh', 0.0)
CENTER_DEV = current_settings.get('center_dev', 0.0)
REG_PARAM = current_settings.get('reg_param', 0.0)
ALPHA = current_settings.get('alpha', 0.0)
MIN_AMP = current_settings.get('min_amp', 0.0)
STATE_INDEX = current_settings.get('state_index', 0)
DEFAULT_CONTOUR = contour_settings.get('selected_contour', 'Gaussian')
NUM_STARTS = current_settings.get('num_starts', 1)
MIN_PEAK_DIST = current_settings.get('min_peak_dist', DEFAULT_MIN_PEAK_DIST)
MARKER_DIAMETER = current_settings.get('marker_diameter', DEFAULT_MARKER_DIAMETER)

ASSIGNMENT_BLOCK_SIZE = 2 * STATE_INDEX

# Read data file using a file dialog
Tk().withdraw()  # Hide the root window
filename = filedialog.askopenfilename(title="CHOOSE RAW DATA FILE")
if not filename:
    print("No file selected. Exiting.")
    exit()

data = np.loadtxt(filename)
x = data[:, 0]
y = data[:, 1]

# Load external data for lower plot
ext_filename = filedialog.askopenfilename(
    title="Select data for lower plot (e.g. simulation)"
)
ext_x, ext_y = None, None
if ext_filename:
    try:
        ext_data = np.loadtxt(ext_filename)
        if ext_data.ndim == 2 and ext_data.shape[1] == 2:
            ext_x = ext_data[:, 0]
            ext_y = ext_data[:, 1] * EXT_SCALE  # Apply external scaling
            print(f"Loaded and scaled (×{EXT_SCALE:.2f}) external data")
        else:
            print("Invalid external data format. Must be 2 columns. Skipping.")
    except Exception as e:
        print(f"Error loading external data: {str(e)}")

# Create figure with two subplots
fig, (ax1, ax3, ax2) = plt.subplots(
    3, 1, 
    sharex=True, 
    figsize=(10, 12),  # Taller figure to accommodate the new plot
    gridspec_kw={'height_ratios': [3, 1, 3]}  # ax1 is 3x tall, ax3 is 1x, ax2 is 2x
)
plt.subplots_adjust(hspace=0.1)

# Plot data on both axes
ax1.plot(x, y, 'b-', label='Raw Data')
ax2.plot(x, y, 'b-', label='Raw Data')

# Add zero lines to both plots
ax1.axhline(0, color='black', linestyle=':', linewidth=0.5, alpha=0.5)
ax2.axhline(0, color='black', linestyle=':', linewidth=0.5, alpha=0.5)

# Plot external data if loaded
if ext_x is not None and ext_y is not None:
    ax2.plot(ext_x, ext_y, 'g-', linewidth=1.2, alpha=0.7, label='External Data')

# ax2.set_title("Simulation (Red=Peaklist, Green=Calculated)")
ax2.set_xlabel("X-axis")
ax1.set_ylabel("Y-axis")
ax2.set_ylabel("Y-axis")

# Setup the new residuals axis
current_ylim = ax1.get_ylim()
ax3.set_ylim(-1.0 * current_ylim[1] / 2, current_ylim[1] / 2)

ax3.set_ylabel("Residual")
ax3.axhline(0, color='black', linestyle=':', linewidth=0.5, alpha=0.5) # Add a zero line for reference

# Calculated peaks handling
def plot_calculated_peaks(filename, ax):
    """
    Plot calculated peaks onto the provided axes, handling STATE_INDEX QNs.
    This now creates pickable v-lines and the persistent, rotated "base" annotations.
    Returns a list of created v-line artists.
    """
    global STATE_INDEX, CALC_SCALE, INTENSITY_THRESH
    global calc_base_annotation_artists

    for artist in calc_base_annotation_artists:
        try: artist.remove()
        except: pass
    calc_base_annotation_artists.clear()
    
    created_vlines = []
    expected_min_cols = 2 + (2 * STATE_INDEX)

    try:
        if ax is None: return []
        xmin_view, xmax_view = ax.get_xlim()

        if not filename or not os.path.exists(filename): return []

        calc_data = np.loadtxt(filename)
        if calc_data.size == 0: return []
        
        if calc_data.ndim == 1:
            if len(calc_data) < expected_min_cols: return []
            calc_data = calc_data.reshape(1, -1)
        elif calc_data.shape[1] < expected_min_cols:
             return []

        y_values = calc_data[:, 1] * CALC_SCALE
        max_intensity = np.max(y_values) if y_values.size > 0 else 1.0
        if max_intensity <= 0: max_intensity = 1.0

        plotted_count = 0
        new_calc_annotations = []
        view_ymin, view_ymax = ax.get_ylim()
        
        for row_idx, row in enumerate(calc_data):
            x_val = row[0]
            if not (xmin_view <= x_val <= xmax_view): continue
            if len(row) < expected_min_cols: continue

            scaled_intensity = row[1] * CALC_SCALE
            if (scaled_intensity / max_intensity) < INTENSITY_THRESH: continue

            lower_qns_str = " ".join([f"{int(qn)}" for qn in row[2:2+STATE_INDEX]])
            upper_qns_str = " ".join([f"{int(qn)}" for qn in row[2+STATE_INDEX:2+(2*STATE_INDEX)]])
            annotation_text = f"{lower_qns_str} → {upper_qns_str}"

            # --- MODIFICATION: Make the v-line itself interactive ---
            vline_artist = ax.vlines(
                x_val, ymin=0, ymax=scaled_intensity,
                colors='green', linewidths=1.0, alpha=0.7,
                # Set a picker with a tolerance of 3 points (makes it easier to hit)
                picker=3 
            )
            
            # Attach the data directly to the vline artist
            # Note: vlines returns a list, so we get the first element
            if isinstance(vline_artist, matplotlib.collections.LineCollection):
                vline_artist.peak_data = {'x': x_val, 'y': scaled_intensity, 'text': annotation_text}

            created_vlines.append(vline_artist)

            if annotation_text:
                text_y_pos = -0.02 * (view_ymax - view_ymin)
                text_artist = ax.text(
                    x_val, text_y_pos, annotation_text,
                    ha='center', va='top', fontsize=6, rotation=90,
                    alpha=0.8, color='black',
                    picker=True
                )
                text_artist.peak_data = {'x': x_val, 'y': scaled_intensity, 'text': annotation_text}
                new_calc_annotations.append(text_artist)

            plotted_count += 1
        
        calc_base_annotation_artists = new_calc_annotations
        print(f"Plotted {plotted_count} calculated peaks from '{filename}' within current view.")

    except Exception as e:
        print(f"ERROR processing calculated peaks file '{filename}': {str(e)}")
        import traceback
        traceback.print_exc()

    # The calling function now receives a list of interactive v-line artists
    return created_vlines

# In the main plotting section, replace previous calculated peaks code with:
calc_filename = filedialog.askopenfilename(title="Choose linelist file")
#if calc_filename:
#    plot_calculated_peaks(calc_filename, ax1)

# Variables for tracking state
dragging = False
start_x, start_y = 0, 0
original_xlim, original_ylim = ax1.get_xlim(), ax1.get_ylim()

# Peak management
peak_coords = []

def generate_simulation_from_peaks(
    peaks_data: list[list[str]],
    x_grid: np.ndarray
) -> np.ndarray:
    """
    Generates a simulated spectrum in memory from a list of peak parameters.

    This function is optimized to calculate a simulation on a given x-axis grid
    without writing to a file. It is used to compute the contribution of
    "background" peaks that lie outside the primary fitting region.

    Args:
        peaks_data (list[list[str]]):
            A list where each inner list contains the string elements of a line
            from peaklist_filename (e.g., ['x', 'y', 'contour', 'fwhm', ...]).
        x_grid (np.ndarray):
            The NumPy array of X-values upon which the simulation will be built.

    Returns:
        np.ndarray:
            A NumPy array of Y-values representing the summed simulation of all
            input peaks, with the same size as x_grid.
    """
    # Pre-allocate the output array for efficiency, initialized to zeros.
    simulated_y = np.zeros_like(x_grid)

    if not peaks_data:
        return simulated_y

    # Get simulation settings
    sim_window_multiplier = current_settings.get('sim_window_fwhm', 5.0)
    if len(x_grid) > 1:
        point_spacing = np.mean(np.diff(x_grid))
    else:
        point_spacing = 1.0 # Fallback

    for line_num, elements in enumerate(peaks_data, 1):
        try:
            center_x = float(elements[1])
            max_y = float(elements[2])
            contour_type = elements[3]
            calc_func = CONTOUR_FUNCTIONS.get(contour_type, calculate_gaussian_value)

            # Dynamic window calculation using the setting
            peak_fwhm = 0.0
            if contour_type == 'Voigt':
                if len(elements) < 6: continue
                fwhm_g, fwhm_l = float(elements[4]), float(elements[5])
                peak_fwhm = max(fwhm_g, fwhm_l) # Use larger component as proxy
                y_window_params = {'fwhm_g': fwhm_g, 'fwhm_l': fwhm_l}
            else: # Gaussian or Lorentzian
                if len(elements) < 5: continue
                peak_fwhm = float(elements[4])
                y_window_params = {'fwhm': peak_fwhm}
            
            window_half_width_units = sim_window_multiplier * peak_fwhm
            window_half_width_points = int(window_half_width_units / point_spacing) + 1

            idx = np.abs(x_grid - center_x).argmin()
            start_idx = max(0, idx - window_half_width_points)
            end_idx = min(len(x_grid), idx + window_half_width_points + 1)
            x_window = x_grid[start_idx:end_idx]

            y_window = calc_func(x=x_window, amplitude=max_y, center=center_x, **y_window_params)
            # --- END MODIFICATION ---

            simulated_y[start_idx:end_idx] += y_window

        except (IndexError, ValueError) as e:
            # Skip any malformed lines in the input data
            print(f"Warning (generate_simulation): Skipping invalid peak data on line {line_num}: {str(e)}")
            continue

    return simulated_y

def _fit_worker(p0_start, x_region, y_region, peaks_info, bounds, conv_tol, alpha, reg_param):
    """
    A single, self-contained unit of work for the parallel fitter.
    This function can be executed in a separate process. It takes a set of
    starting parameters and runs one minimization.
    """
    # The objective function is defined inside the worker to ensure it has
    # access to all necessary variables from the arguments.
    def objective_function(params, x_data, y_data, peaks_info_struct):
        y_pred = np.zeros_like(x_data)
        param_idx = 0
        
        for peak_struct in peaks_info_struct:
            contour_type = peak_struct[3]
            calc_func = CONTOUR_FUNCTIONS[contour_type]
            
            amp = params[param_idx]
            center = params[param_idx + 1]
            
            if contour_type == 'Voigt':
                fwhm_g = params[param_idx + 2]
                fwhm_l = params[param_idx + 3]
                y_pred += calc_func(x=x_data, amplitude=amp, center=center, fwhm_g=fwhm_g, fwhm_l=fwhm_l)
                param_idx += 4
            else:
                fwhm = params[param_idx + 2]
                y_pred += calc_func(x=x_data, amplitude=amp, center=center, fwhm=fwhm)
                param_idx += 3
        
        # Use the alpha and reg_param passed into the worker
        return calculate_hybrid_rss(y_data, y_pred, x_data, alpha, len(peaks_info_struct))

    # Run the optimization. disp=False is important for clean parallel logs.
    result = minimize(
        fun=objective_function,
        x0=p0_start,
        args=(x_region, y_region, peaks_info),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 25000, 'ftol': conv_tol}
    )
    
    # Return the entire result object
    return result

def fit_peaks(data_file, limits_file, peaklist_file, min_amplitude_inp, num_starts, conv_tol):
    """
    Fit peaks using a simultaneous optimization of all parameters.
    This version accounts for the influence of VISIBLE peaks outside the
    fitting region by simulating their contribution and subtracting it from the
    experimental data before fitting. This is much faster for large peaklists
    when the user is zoomed in.
    
    MODIFIED: Incorporates an iterative validation loop to remove fitted peaks
    that are closer than the user-defined MIN_PEAK_DIST.
    """
    global contour_settings, CENTER_DEV, ALPHA, REG_PARAM, peak_coords, MIN_PEAK_DIST

    # --- Phase 1: Initialization and Data Loading (Outer part) ---
    if not os.path.exists(data_file):
        print(f"{data_file} not found.")
        return
    data = np.loadtxt(data_file)
    x, y = data[:, 0], data[:, 1]

    with open(limits_file, "r") as f:
        lines = [line for line in f if line.strip()]
    if len(lines) < 2:
        print(f"{limits_file} must contain at least two lines.")
        return
    x1, x2 = sorted([float(lines[0].strip()), float(lines[1].strip())])
    
    try:
        with open("simlim.txt", "r") as f_sim:
            lines = [line for line in f_sim if line.strip()]
        if len(lines) < 2: raise ValueError("simlim.txt does not contain two valid limits.")
        view_xmin, view_xmax = sorted([float(lines[0].strip()), float(lines[1].strip())])
        print(f"Using simulation limits from simlim.txt: [{view_xmin:.4f}, {view_xmax:.4f}]")
    except (FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not read simlim.txt ({e}). Disabling background peak simulation.")
        view_xmin, view_xmax = np.inf, -np.inf

    # --- NEW: Outer Iteration Loop for Validation ---
    max_fit_iterations = 10
    for iteration in range(max_fit_iterations):
        print(f"\n--- Fitting Iteration #{iteration + 1} ---")

        # --- Phase 1b: Reread Peaklist on each iteration ---
        # This is CRITICAL as the file may have been modified by the previous iteration's validation step.
        with open(peaklist_file, "r") as f:
            all_lines = f.readlines()

        # --- Phase 2: Peak Partitioning and Background Subtraction ---
        peaks_to_fit_info = []
        background_peaks_data = []
        for idx, line in enumerate(all_lines):
            parts = line.strip().split()
            if len(parts) < 4: continue
            try:
                peak_x = float(parts[1])
                contour_type = parts[3]
                if contour_type not in CONTOUR_FUNCTIONS: contour_type = 'Gaussian'
                if x1 <= peak_x <= x2:
                    peaks_to_fit_info.append([idx, peak_x, float(parts[2]), contour_type, parts])
                elif view_xmin <= peak_x <= view_xmax:
                    background_peaks_data.append(parts)
            except (ValueError, IndexError):
                continue
        
        if not peaks_to_fit_info:
            print("No peaks to fit in the specified region. Stopping.")
            return 0.0

        if background_peaks_data:
            print(f"Accounting for background: Simulating {len(background_peaks_data)} visible peaks outside the fit region.")
            background_simulation = generate_simulation_from_peaks(background_peaks_data, x)
            y_corrected = y - background_simulation
        else:
            y_corrected = y.copy()
            print("No visible background peaks outside the fit region to account for.")

        region_mask = (x >= x1) & (x <= x2)
        x_region, y_region = x[region_mask], y_corrected[region_mask]
        if x_region.size == 0:
            print("No data in selected region.")
            return 0.0
        min_amplitude = min_amplitude_inp

        # --- Phase 3 & 4: Build Model and Prepare for Multi-Start (CORRECTED) ---
        p0 = []
        bounds = []
        variable_indices = []
        variable_bounds = []
        param_index_counter = 0

        for peak in peaks_to_fit_info:
            initial_center, initial_amp, contour_type, original_parts = peak[1], peak[2], peak[3], peak[4]
            symbol = original_parts[0]
            if symbol == 'M':
                print(f"  - Fixing parameters for marked peak at {initial_center:.4f}")
                p0.append(initial_amp); bounds.append((initial_amp, initial_amp)); param_index_counter += 1
                p0.append(initial_center); bounds.append((initial_center, initial_center)); param_index_counter += 1
                if contour_type == 'Voigt':
                    fwhm_g_val, fwhm_l_val = float(original_parts[4]), float(original_parts[5])
                    p0.extend([fwhm_g_val, fwhm_l_val]); bounds.extend([(fwhm_g_val, fwhm_g_val), (fwhm_l_val, fwhm_l_val)]); param_index_counter += 2
                else:
                    fwhm_val = float(original_parts[4])
                    p0.append(fwhm_val); bounds.append((fwhm_val, fwhm_val)); param_index_counter += 1
            else:
                contour_params = contour_settings['contours'][contour_type]
                amplitude_var_percent = contour_params.get('amplitude_var', 100.0)
                amp_b = (max(0, initial_amp - initial_amp * (amplitude_var_percent / 100.0)), initial_amp + initial_amp * (amplitude_var_percent / 100.0))
                p0.append(initial_amp); bounds.append(amp_b)
                if amp_b[0] < amp_b[1]: variable_indices.append(param_index_counter); variable_bounds.append(amp_b)
                param_index_counter += 1
                center_b = (initial_center - CENTER_DEV, initial_center + CENTER_DEV)
                p0.append(initial_center); bounds.append(center_b)
                if center_b[0] < center_b[1]: variable_indices.append(param_index_counter); variable_bounds.append(center_b)
                param_index_counter += 1
                if contour_type == 'Voigt':
                    fwhm_g_val, fwhm_l_val = contour_params.get('fwhm_g', DEFAULT_FWHM_G), contour_params.get('fwhm_l', DEFAULT_FWHM_L)
                    var_g, var_l = contour_params.get('fwhm_g_var', DEFAULT_FWHM_G_VAR), contour_params.get('fwhm_l_var', DEFAULT_FWHM_L_VAR)
                    fwhm_g_b = (fwhm_g_val * (1 - var_g / 100), fwhm_g_val * (1 + var_g / 100))
                    fwhm_l_b = (fwhm_l_val * (1 - var_l / 100), fwhm_l_val * (1 + var_l / 100))
                    p0.extend([fwhm_g_val, fwhm_l_val]); bounds.extend([fwhm_g_b, fwhm_l_b])
                    if fwhm_g_b[0] < fwhm_g_b[1]: variable_indices.append(param_index_counter); variable_bounds.append(fwhm_g_b)
                    param_index_counter += 1
                    if fwhm_l_b[0] < fwhm_l_b[1]: variable_indices.append(param_index_counter); variable_bounds.append(fwhm_l_b)
                    param_index_counter += 1
                else:
                    fwhm_val, var = contour_params.get('fwhm', DEFAULT_FWHM), contour_params.get('fwhm_var', DEFAULT_FWHM_VAR)
                    fwhm_b = (fwhm_val * (1 - var / 100), fwhm_val * (1 + var / 100))
                    p0.append(fwhm_val); bounds.append(fwhm_b)
                    if fwhm_b[0] < fwhm_b[1]: variable_indices.append(param_index_counter); variable_bounds.append(fwhm_b)
                    param_index_counter += 1
        p0 = np.array(p0)

        # --- Phase 5a: Generate Starting Parameters ---
        all_p0_starts = [p0]
        if num_starts > 1 and variable_indices:
            print(f"Generating {num_starts - 1} additional starting parameter sets for {len(variable_indices)} variable parameters.")
            l_bounds_var = np.array([b[0] for b in variable_bounds]); u_bounds_var = np.array([b[1] for b in variable_bounds])
            sampler = qmc.Sobol(d=len(variable_indices), scramble=True); sample_points = sampler.random(n=num_starts - 1)
            scaled_points = qmc.scale(sample_points, l_bounds_var, u_bounds_var)
            for i in range(num_starts - 1):
                new_p0 = p0.copy()
                for j, var_idx in enumerate(variable_indices): new_p0[var_idx] = scaled_points[i, j]
                all_p0_starts.append(new_p0)
        elif num_starts > 1:
            print("Multi-start requested, but all parameters in the fit region are fixed. Using only one start.")
        
        # --- Phase 5b: Parallel Optimization ---
        print(f"Starting parallel fit for {len(peaks_to_fit_info)} peaks across {len(all_p0_starts)} attempts...")
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(_fit_worker)(start_params, x_region, y_region, peaks_to_fit_info, bounds, conv_tol, ALPHA, REG_PARAM)
            for start_params in all_p0_starts)
        best_result, best_rss = None, np.inf
        for res in results:
            if res.success and res.fun < best_rss: best_rss, best_result = res.fun, res
        if best_result is None:
            print("Warning: Parallel fitting failed to find a converged solution. Using best available result."); best_result = min(results, key=lambda r: r.fun)
        print(f"Parallel fitting complete. Best RSS found: {best_result.fun:.6e}")
        
        # --- Phase 6: Result Processing and Pruning ---
        p_opt = best_result.x; final_rss = best_result.fun
        kept_peaks = {}
        valid_peaks_info = [] # Store final info for validation
        
        param_idx = 0
        for peak_info_item in peaks_to_fit_info:
            line_idx, _, _, contour_type, original_parts = peak_info_item
            new_amp, new_center = p_opt[param_idx], p_opt[param_idx+1]
            if contour_type == 'Voigt':
                new_fwhm_params = (p_opt[param_idx+2], p_opt[param_idx+3]); param_idx += 4
            else:
                new_fwhm_params = (p_opt[param_idx+2],); param_idx += 3
            
            if new_amp >= min_amplitude:
                kept_peaks[line_idx] = (new_center, new_amp, contour_type, new_fwhm_params, original_parts)
                # If it was a fitted peak, add it to the list for validation
                if original_parts[0] == 'S':
                    valid_peaks_info.append({'line_idx': line_idx, 'center': new_center, 'amplitude': new_amp})

        if len(kept_peaks) < len(peaks_to_fit_info):
            print(f"Pruning: Removing {len(peaks_to_fit_info) - len(kept_peaks)} peaks with amplitude < {min_amplitude}")

        # --- NEW: Phase 6b: Iterative Validation based on Minimal Peak Distance ---
        if MIN_PEAK_DIST <= 0:
            print("Minimal peak distance validation is disabled (distance <= 0).")
            break # Exit the validation loop and proceed to final save

        # Build a comprehensive list of all peaks in the wider simlim region with their FINAL positions
        all_peaks_in_simlim = []
        for idx, line in enumerate(all_lines):
            parts = line.strip().split()
            if len(parts) < 2: continue
            try:
                peak_x = float(parts[1])
                if view_xmin <= peak_x <= view_xmax:
                    # If this peak was just fitted, use its NEW position. Otherwise, use its old one.
                    if idx in kept_peaks:
                        final_center = kept_peaks[idx][0]
                        all_peaks_in_simlim.append({'line_idx': idx, 'center': final_center})
                    else:
                        all_peaks_in_simlim.append({'line_idx': idx, 'center': peak_x})
            except ValueError:
                continue

        indices_to_remove = set()
        # Validate each fitted 'S' peak against all other peaks in the simulation window
        for peak_to_validate in valid_peaks_info:
            is_valid = True
            for other_peak in all_peaks_in_simlim:
                if peak_to_validate['line_idx'] == other_peak['line_idx']:
                    continue # Don't compare a peak to itself
                
                distance = abs(peak_to_validate['center'] - other_peak['center'])
                if distance < MIN_PEAK_DIST:
                    # This peak is too close to another. Mark it for removal.
                    # We will remove the peak with the smaller amplitude between the two.
                    other_peak_amp = next((p['amplitude'] for p in valid_peaks_info if p['line_idx'] == other_peak['line_idx']), float('inf'))
                    if peak_to_validate['amplitude'] < other_peak_amp:
                         indices_to_remove.add(peak_to_validate['line_idx'])
                    # If amplitudes are identical, this logic implicitly keeps the one with the lower index
                    break # Move to the next peak to validate
        
        if indices_to_remove:
            print(f"Validation failed: Found {len(indices_to_remove)} peak(s) violating the minimal distance of {MIN_PEAK_DIST}.")
            
            # Create a new list of lines, excluding the ones to be removed
            new_all_lines = [line for i, line in enumerate(all_lines) if i not in indices_to_remove]
            
            # Save the reduced peaklist and continue to the next iteration
            try:
                safe_write_to_file(peaklist_file, new_all_lines)
                print(f"Removed invalid peaks and saved updated {peaklist_file}. Refitting...")
                continue # Go to the next iteration of the fitting loop
            except IOError as e:
                print(f"CRITICAL ERROR: Could not save updated peak list. Aborting fit. Error: {e}")
                return final_rss # Abort with the current RSS
        else:
            print("Validation successful: All fitted peaks satisfy the minimal distance criterion.")
            break # Exit the validation loop, fit is stable

    else: # This 'else' belongs to the 'for' loop
        print(f"WARNING: Fit did not stabilize after {max_fit_iterations} iterations. Using the current result.")

    # --- Phase 7: Finalization and File Update ---
    # This block now runs only after the validation loop has successfully completed or timed out.
    output_lines = []
    for line_idx, line in enumerate(all_lines):
        if line_idx in kept_peaks:
            new_center, new_amp, contour_type, new_fwhm_params, original_parts = kept_peaks[line_idx]
            
            # Calculate new integral
            if contour_type == 'Voigt':
                fwhm_g, fwhm_l = new_fwhm_params; sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2))); gamma = fwhm_l / 2.0
                scaling_factor = voigt_profile(0, sigma, gamma); new_integral = new_amp / scaling_factor if scaling_factor > 1e-10 else 0.0
            elif contour_type == 'Lorentzian':
                new_integral = new_amp * new_fwhm_params[0] * (np.pi / 2.0)
            else: # Gaussian
                new_integral = new_amp * new_fwhm_params[0] * np.sqrt(np.pi) / (2 * np.sqrt(np.log(2)))

            new_line_parts = [original_parts[0], f"{new_center:.6f}", f"{new_amp:.6f}", contour_type]
            new_line_parts.extend([f"{f:.6e}" for f in new_fwhm_params])
            new_line_parts.append(f"{new_integral:.6e}")
            assignment_start_index = 7 if contour_type == 'Voigt' else 6
            extra = " ".join(original_parts[assignment_start_index:]) if len(original_parts) > assignment_start_index else ""
            new_line = " ".join(new_line_parts) + (f" {extra}" if extra else "") + "\n"
            output_lines.append(new_line)
        elif line_idx in {p[0] for p in peaks_to_fit_info}:
            pass # This was a fitted peak that got pruned, so don't append it
        else:
            output_lines.append(line) # This peak was not part of the fit, keep it as is
            
    if output_lines and not output_lines[-1].endswith('\n'): output_lines[-1] += '\n'
    try:
        safe_write_to_file(peaklist_file, output_lines)
    except IOError:
        print(f"Error: Could not update the peak list file. Fit results were not saved.")
        return final_rss
    print(f"Updated peaks saved to {peaklist_file}. Final peak count in region: {len(kept_peaks)}")
    print(f"Final Best RSS: {final_rss:.6e}")

    return final_rss
    
def filter_file_by_range(input_file_name: str, lower_limit: float, upper_limit: float, output_file_name: str) -> int:
    """
    Scans an input file and copies lines to an output file if the first value
    on a line falls within a specified numerical range.

    This function is designed to be robust, handling potential errors such as
    missing files, non-numeric data, and empty lines gracefully.

    Args:
        input_file_name (str): The path to the source text file.
        lower_limit (float): The lower boundary of the range (inclusive).
        upper_limit (float): The upper boundary of the range (inclusive).
        output_file_name (str): The path to the destination text file.
                                It will be created or overwritten.

    Returns:
        int: The number of lines written to the output file. Returns -1 if
             a file-level error occurred (e.g., input file not found).
    """
    # Ensure the limits are in the correct order for comparison
    if lower_limit > upper_limit:
        lower_limit, upper_limit = upper_limit, lower_limit
        print(f"Warning: Lower limit was greater than upper limit. Swapped to [{lower_limit}, {upper_limit}].")

    lines_written_count = 0

    try:
        # Use 'with' statements for safe file handling (ensures files are closed)
        with open(input_file_name, 'r') as infile, open(output_file_name, 'w') as outfile:
            
            # Iterate through each line in the input file
            for line in infile:
                # Strip leading/trailing whitespace and split the line into parts
                parts = line.strip().split()

                # --- Data Validation ---
                # Skip empty lines or lines without any data
                if not parts:
                    continue

                try:
                    # Attempt to convert the first part to a float
                    first_value = float(parts[1])
                    
                    # --- The Core Logic ---
                    # Check if the value is within the specified range
                    if lower_limit <= first_value <= upper_limit:
                        # If it is, write the original, unmodified line to the output file
                        outfile.write(line)
                        lines_written_count += 1

                except (ValueError, IndexError):
                    # This block catches two potential issues:
                    # 1. ValueError: If parts[0] is not a valid number (e.g., "header").
                    # 2. IndexError: Should not happen due to the 'if not parts' check, but is safe to include.
                    # We'll just skip these malformed lines and continue.
                    # You could add a print statement here if you want to be notified of skipped lines.
                    # print(f"Skipping malformed line: {line.strip()}")
                    continue

        print(f"Processing complete. Wrote {lines_written_count} lines to '{output_file_name}'.")
        return lines_written_count

    except FileNotFoundError:
        print(f"Error: The input file '{input_file_name}' was not found.")
        return -1
    except IOError as e:
        print(f"Error: Could not read from or write to a file. Details: {e}")
        return -1

def calculate_contour_simulation(input_file):
    """
    Calculate contour simulations using a memory-efficient, pre-allocated array method.
    This version avoids creating large intermediate lists and expensive sorting operations.
    """
    print(f"Beginning contour simulation")
    try:
        # Load the global X-axis data from the experimental spectrum.
        # This is the grid upon which the simulation will be built.
        original_x = data[:, 0]
        
        # 1. PRE-ALLOCATION
        # Create the final simulation Y-array, initialized to zeros. This is the key optimization.
        simulated_y = np.zeros_like(original_x)

        # Get simulation settings
        sim_window_multiplier = current_settings.get('sim_window_fwhm', 5.0)
        if len(original_x) > 1:
            point_spacing = np.mean(np.diff(original_x))
        else:
            point_spacing = 1.0 # Fallback

        # Load peak data from the specified file
        if not os.path.exists(input_file):
            print(f"Info: {input_file} not found for simulation. Clearing Autosim.txt.")
            if os.path.exists("Autosim.txt"): open("Autosim.txt", "w").close()
            return
            
        with open(input_file, "r") as f:
            peak_count = 0
            for line_num, line in enumerate(f, 1):
                elements = line.strip().split()
                if len(elements) < 4: continue # Skip invalid lines

                try:
                    center_x = float(elements[1])
                    max_y = float(elements[2])
                    contour_type = elements[3]
                    
                    # Dynamic window calculation using the setting
                    peak_fwhm = 0.0
                    if contour_type == 'Voigt':
                        if len(elements) < 6: continue
                        fwhm_g, fwhm_l = float(elements[4]), float(elements[5])
                        peak_fwhm = max(fwhm_g, fwhm_l)
                        fwhm_params = (fwhm_g, fwhm_l)
                    else:
                        peak_fwhm = float(elements[4])
                        fwhm_params = peak_fwhm
                    
                    peak_count += 1
                    
                    window_half_width_units = sim_window_multiplier * peak_fwhm
                    window_half_width_points = int(window_half_width_units / point_spacing) + 1

                    idx = np.abs(original_x - center_x).argmin()
                    
                    start_idx = max(0, idx - window_half_width_points) 
                    end_idx = min(len(original_x), idx + window_half_width_points + 1)
                    
                    # Get the X-values for just this window
                    x_window = original_x[start_idx:end_idx]
                    
                    # Get the appropriate calculation function (Gaussian, Lorentzian, Voigt)
                    calc_func = CONTOUR_FUNCTIONS.get(contour_type, calculate_gaussian_value)
                    
                    # Calculate the Y-values for the window
                    if contour_type == 'Voigt':
                        fwhm_g, fwhm_l = fwhm_params
                        y_window = calc_func(x=x_window, amplitude=max_y, center=center_x, fwhm_g=fwhm_g, fwhm_l=fwhm_l)
                    else:
                        y_window = calc_func(x=x_window, amplitude=max_y, center=center_x, fwhm=fwhm_params)
                    
                    # 3. --- IN-PLACE SUMMATION ---
                    # Add the calculated window directly into the final array.
                    # This replaces list-appending, converting, sorting, and summing.
                    simulated_y[start_idx:end_idx] += y_window

                except (IndexError, ValueError) as e:
                    print(f"Skipping invalid line {line_num} in {input_file}: {str(e)}")
                    continue
        
        # If no peaks were found or the file was empty, clear the output and exit
        if peak_count == 0:
            print(f"No valid peaks found in {input_file} for simulation.")
            if os.path.exists("Autosim.txt"): open("Autosim.txt", "w").close()
            return

        # 4. --- SAVE FINAL RESULT ---
        # The data is already sorted and summed correctly. Just save it.
        final_data = np.column_stack((original_x, simulated_y))
        np.savetxt("Autosim.txt", final_data, fmt="%.6f %.6f")

        print(f"Simulated {peak_count} peaks from {input_file} using specified contours.")

    except Exception as e:
        import traceback
        print(f"CRITICAL Simulation error using {input_file}: {str(e)}")
        traceback.print_exc()

def save_peak(x_val, y_val):
    """
    Save a peak to file and update the single plot object using robust rebuild.
    """
    global peak_plot_object, peak_coords, peak_data_store, ax1, fig
    global DEFAULT_FWHM, DEFAULT_FWHM_G, DEFAULT_FWHM_L, DEFAULT_CONTOUR

    # --- Dynamically calculate integral based on the default contour type ---
    if DEFAULT_CONTOUR == 'Lorentzian':
        integral = y_val * DEFAULT_FWHM * (np.pi / 2.0)
        new_line_content = f"S {x_val:.6f} {y_val:.6e} {DEFAULT_CONTOUR} {DEFAULT_FWHM:.4e} {integral:.6e}"
    elif DEFAULT_CONTOUR == 'Voigt':
        # The integral of the 'calculate_voigt_value' function is amplitude / scaling_factor
        sigma = DEFAULT_FWHM_G / (2 * np.sqrt(2 * np.log(2)))
        gamma = DEFAULT_FWHM_L / 2.0
        scaling_factor = voigt_profile(0, sigma, gamma)
        integral = y_val / scaling_factor if scaling_factor > 1e-10 else 0.0
        new_line_content = f"S {x_val:.6f} {y_val:.6e} {DEFAULT_CONTOUR} {DEFAULT_FWHM_G:.4e} {DEFAULT_FWHM_L:.4e} {integral:.6e}"
    else:  # Default to Gaussian
        integral = y_val * DEFAULT_FWHM * np.sqrt(np.pi) / (2 * np.sqrt(np.log(2)))
        new_line_content = f"S {x_val:.6f} {y_val:.6e} {DEFAULT_CONTOUR} {DEFAULT_FWHM:.4e} {integral:.6e}"

    new_line_file = new_line_content + "\n" # For file writing
# --- 1. Append to peaklist_filename ---
    try:
        # *** USE "a+" mode for read/append ***
        with open(peaklist_filename, "a+") as f:
            # Ensure the file ends with a newline BEFORE appending if it doesn't
            f.seek(0, os.SEEK_END) # Go to end of file
            file_pos = f.tell()
            if file_pos > 0: # Check if file is not empty
                 f.seek(file_pos - 1, os.SEEK_SET) # Go to last character's position
                 last_char = f.read(1) # Read the last character - NOW ALLOWED
                 if last_char != '\n':
                      # If missing newline, need to append it first.
                      # Since we are already at the end after the read,
                      # we can just write the newline and then the new content.
                      f.write('\n')
                      print("Warning (save_peak): Added missing newline before appending to Peaklist.")
            # Whether newline was added or not, f is now positioned at the very end
            # Now append the new line content
            f.write(new_line_file)
    except FileNotFoundError:
        # Handle case where file doesn't exist yet (a+ creates it)
         with open(peaklist_filename, "w") as f: # Create and write the first line
            f.write(new_line_file)
         print("Created Peaklist and saved first peak.")
    except Exception as e:
        print(f"Error saving peak to Peaklist file: {str(e)}")
        return # Don't update plot if save failed

    # --- 2. Append to in-memory lists ---
    peak_coords.append((x_val, y_val))
    # Store the elements as they appear in the file line (without newline)
    new_elements = new_line_content.split()
    peak_data_store.append(new_elements)

# --- 3. Trigger a full, robust redraw of all peaks ---
    # This single call replaces all the direct plot manipulation.
    # It ensures that all peak styles are correctly applied every time.
    load_existing_peaks(peaklist_filename)

    # --- 4. Redraw the canvas ---
    if fig and fig.canvas:
        try:
            fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error during draw_idle in save_peak: {e}")
            
    print(f"Saved and plotted point: ({x_val:.5f}, {y_val:.5f})")

def remove_peak(x_val, y_val):
    """
    Finds a peak with coordinates EXACTLY matching the input arguments and
    removes it from the file, plot object data, and annotations.
    If no exact match is found, the function does nothing.
    """
    global peak_plot_object, peak_coords, peak_data_store, ax1, fig

    # --- 1. Search for the peak with EXACTLY matching coordinates ---
    # We use np.isclose for safe floating-point comparison.
    # Initialize a "not found" index.
    idx_to_remove = -1
    for i, (px, py) in enumerate(peak_coords):
        # Check if both x and y coordinates are very close to the target values
        if np.isclose(px, x_val) and np.isclose(py, y_val):
            idx_to_remove = i  # We found our match
            break              # Exit the loop since we found the first match

    # --- 2. If no matching peak was found, exit the function ---
    if idx_to_remove == -1:
        # You can uncomment the following line for debugging if you want.
        # print(f"Info: No peak found with exact coordinates ({x_val:.6f}, {y_val:.6f}).")
        return # Do nothing, as requested.

    # --- 3. If a match was found, proceed with removal ---
    # Store the coordinates for later use in annotation removal
    removed_px, removed_py = peak_coords[idx_to_remove]

    # --- A. Update the Peaklist file (with robustness) ---
    try:
        with open(peaklist_filename, "r") as f:
            lines = f.readlines()

        # Check if the found index is valid for the lines list
        if idx_to_remove < len(lines):
            # Remove the line at the specific index
            del lines[idx_to_remove]

            # Use the safe write function to prevent data corruption
            safe_write_to_file(peaklist_filename, lines)
        else:
            print(f"Error: Index mismatch. Found peak at index {idx_to_remove} but Peaklist only has {len(lines)} lines.")
            return # Abort if data is inconsistent

    except FileNotFoundError:
        print("Error: Peaklist not found. Cannot remove peak from file.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during file update: {str(e)}")
        return

    # --- B. Remove from in-memory lists *after* successful file update ---
    del peak_coords[idx_to_remove]
    del peak_data_store[idx_to_remove]

# --- C. Trigger a full, robust redraw of all peaks and annotations ---
    # This single call handles removing the old plot objects (markers and annotations)
    # and creating new ones that are perfectly in sync with the updated database.
    load_existing_peaks(peaklist_filename)

    # --- D. Redraw the canvas to show all changes ---
    if fig and fig.canvas:
        fig.canvas.draw_idle()

            
def load_existing_peaks(filename):
    """
    Load existing peaks from file, filter by current X-axis view,
    plot them efficiently as a single object on BOTH ax1 and ax2,
    and manage their respective annotations.
    """
    global peak_plot_object, peak_plot_object_ax2
    global peak_coords, peak_data_store
    global base_annotation_artists, peaklist_annotation_artists_ax2
    global ax1, ax2, fig, STATE_INDEX, ASSIGNMENT_BLOCK_SIZE

    # --- 1. Clear previous PLOT ELEMENTS ---
    peak_coords.clear()
    peak_data_store.clear()

    if peak_plot_object:
        try: peak_plot_object.remove()
        except: pass
        peak_plot_object = None

    if peak_plot_object_ax2:
        try: peak_plot_object_ax2.remove()
        except: pass
        peak_plot_object_ax2 = None

    for artist in base_annotation_artists:
        try: artist.remove()
        except: pass
    base_annotation_artists.clear()

    for artist in peaklist_annotation_artists_ax2:
        try: artist.remove()
        except: pass
    peaklist_annotation_artists_ax2.clear()

    # --- 2. Read ALL data from file into global stores ---
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    elements = line.strip().split()
                    if len(elements) < 3: continue
                    try:
                        xp, yp = float(elements[1]), float(elements[2])
                        peak_coords.append((xp, yp))
                        peak_data_store.append(elements)
                    except ValueError:
                        print(f"Warning: Skipping line {line_num} in {filename}")
                        continue
        
        # --- [Sections 3, 4, and 5 remain the same] ---
        try:
            xmin_view, xmax_view = ax1.get_xlim()
            limits_valid = True
            if xmax_view <= xmin_view: limits_valid = False
        except Exception:
            limits_valid = False; xmin_view, xmax_view = -np.inf, np.inf

        plot_xp, plot_yp, visible_peak_indices = [], [], []
        plot_colors, plot_edgecolors, plot_linewidths = [], [], [] # Face color, edge color, edge width

        if peak_coords:
            for i, (px, py) in enumerate(peak_coords):
                if not limits_valid or (xmin_view <= px <= xmax_view):
                    # Add coordinates to the plot lists
                    plot_xp.append(px)
                    plot_yp.append(py)
                    visible_peak_indices.append(i)
                    
                    # --- [NEW] Advanced Logic for per-point styling ---
                    symbol = peak_data_store[i][0]
                    if symbol == 'M':
                        # Style for MARKED peaks
                        plot_colors.append('magenta')
                        plot_edgecolors.append('black')
                        plot_linewidths.append(0.6) # A thin but visible edge
                    else:
                        # Style for STANDARD peaks
                        plot_colors.append('red')
                        plot_edgecolors.append('none') # 'none' means no edge color
                        plot_linewidths.append(0)      # No edge width
        
        # --- [MODIFIED] Use all the style lists when plotting ---
        if plot_xp and ax1:
            peak_plot_object = ax1.scatter(plot_xp, plot_yp, marker='.', 
                                           color=plot_colors,
                                           edgecolors=plot_edgecolors,
                                           linewidths=plot_linewidths,
                                           s=MARKER_DIAMETER**2,  # <-- CHANGE THIS LINE
                                           alpha=0.8, zorder=3, 
                                           label='Peaks (Visible Ax1)',
                                           picker=5)
        if plot_xp and ax2:
            peak_plot_object_ax2 = ax2.scatter(plot_xp, plot_yp, marker='.', 
                                               color=plot_colors,
                                               edgecolors=plot_edgecolors,
                                               linewidths=plot_linewidths,
                                               s=MARKER_DIAMETER**2,  # <-- AND CHANGE THIS LINE
                                               alpha=0.8, zorder=3, 
                                               label='Peaks (Visible Ax2)')
        
        # --- 6. Plot Annotations and make them INTERACTIVE ---
        new_base_annotations = []
        if visible_peak_indices:
            for i in visible_peak_indices:
                 xp, yp = peak_coords[i]
                 elements = peak_data_store[i]
                 expected_y_pos = yp * 1.1

                 # Determine where assignments start based on contour type ---
                 contour_type = elements[3]
                 if contour_type == 'Voigt':
                     # X, Y, Contour, FWHM_G, FWHM_L, Integral -> QNs start at index 6
                     assignment_start_index = 7
                 else:
                     # X, Y, Contour, FWHM, Integral -> QNs start at index 5
                     assignment_start_index = 6
                 
                 if len(elements) > assignment_start_index:
                     assignment_numbers_str = elements[assignment_start_index:]
                 else:
                     assignment_numbers_str = []
                 # --- END FIX ---
                 if assignment_numbers_str:
                     annotation_lines = []
                     for k in range(0, len(assignment_numbers_str), ASSIGNMENT_BLOCK_SIZE):
                         chunk = assignment_numbers_str[k:k+ASSIGNMENT_BLOCK_SIZE]
                         if len(chunk) == ASSIGNMENT_BLOCK_SIZE:
                             lower = " ".join(chunk[0:STATE_INDEX])
                             upper = " ".join(chunk[STATE_INDEX:ASSIGNMENT_BLOCK_SIZE])
                             annotation_lines.append(f"{lower} → {upper}")
                     
                     if annotation_lines:
                         annotation_text = "\n".join(annotation_lines)
                         text_artist = ax1.text(
                             xp, expected_y_pos, annotation_text,
                             ha='center', va='bottom', fontsize=6, rotation=90,
                             alpha=0.7, color='black', linespacing=1.2, zorder=5,
                             # --- THIS IS THE CRITICAL FIX ---
                             picker=True
                         )
                         # --- THIS IS A ROBUSTNESS IMPROVEMENT ---
                         # Store the peak's master index directly on the artist object
                         text_artist.peak_index = i
                         new_base_annotations.append(text_artist)

        base_annotation_artists = new_base_annotations

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR in load_existing_peaks: {e}")

def calculate_hybrid_rss(y_true, y_pred, x_vals, alpha, n_peaks):
    """Calculate hybrid residual incorporating both contour and integral errors"""
    
    contour_rss = np.sum((y_true - y_pred) ** 2)
    
    if alpha > 0.0:
        # Calculate integral residuals using trapezoidal rule
        integral_true = np.trapezoid(y_true, x_vals)
        integral_pred = np.trapezoid(y_pred, x_vals)
        integral_rss = np.abs(integral_true - integral_pred)
    else:
        integral_rss = 0.0
    
    return (1 - alpha) * contour_rss + alpha * integral_rss + REG_PARAM * n_peaks

# Event handlers
def on_press(event):
    global dragging, start_x, start_y, original_xlim, original_ylim

    # 1. Handle Right-Click to open the peak editor
    # We check if the button is the right mouse button and if it's a single click.
    if event.button == MouseButton.RIGHT and event.inaxes == ax1 and not event.dblclick:
        handle_pick(event)  # This is the function that opens the assignment window
        return              # CRITICAL: This stops the event and prevents the default zoom

    # 2. Handle Left-Click for panning (original logic)
    if event.button == MouseButton.LEFT and event.inaxes == ax1:
        dragging = True
        start_x = event.xdata
        start_y = event.ydata
        original_xlim = ax1.get_xlim()
        original_ylim = ax1.get_ylim()

def on_motion(event):
    global dragging
    if not dragging:
        return

    if dragging and event.inaxes == ax1 and event.button == 1:
        current_x = event.xdata
        current_y = event.ydata
        if None in (current_x, current_y):
            return
        dx = current_x - start_x
        dy = current_y - start_y
        
        # Calculate new limits
        new_xlim = (original_xlim[0] - dx, original_xlim[1] - dx)
        new_ylim = (original_ylim[0] - dy, original_ylim[1] - dy)
        
        # Update BOTH axes' X-limits explicitly
        ax1.set_xlim(new_xlim)
        ax2.set_xlim(new_xlim)  # This line forces synchronization
        
        # Update BOTH axes' Y-limits explicitly
        ax1.set_ylim(new_ylim)
        ax2.set_ylim(new_ylim)
        
        plt.draw()

def on_release(event):
    global dragging
    if event.button == 1:
        # Check if dragging actually occurred before updating
        if dragging:
             _update_plots_for_view()
        dragging = False

def on_key(event):
    
    global ax1, ax2, ax3, fig, fit_lines, filename, MIN_AMP, NUM_STARTS, peak_coords, y
    
    update_needed = False # Flag to call update once at the end

    # Note: the check "if event.inaxes != ax1:" is okay because ax1 is already global.
    # However, for consistency and clarity, we include it in the global list.
    if event.inaxes not in [ax1, ax3]: # Allow events from main plot OR residual plot
        return

    key = event.key.lower()

    # Note: the check "if event.inaxes != ax1:" is okay because ax1 is already global.
    # However, for consistency and clarity, we include it in the global list.
    if event.inaxes not in [ax1, ax3]: # Allow events from main plot OR residual plot
        return

    key = event.key.lower()

    if key == 't':
        cursor_x = event.xdata
        cursor_y = event.ydata
        if cursor_x is None: return

        # Check which subplot the key was pressed in
        if event.inaxes == ax1:
            # User clicked on the main plot
            print(f"Tagging peak from main plot at X={cursor_x}, Y={cursor_y}")
            save_peak(cursor_x, cursor_y)
        elif event.inaxes == ax3:
            # User clicked on the residual plot
            print(f"Tagging new peak from RESIDUAL plot at X={cursor_x}, Amplitude={cursor_y}")
            save_peak(cursor_x, cursor_y)
        
    elif key == 'c':
        # Get the click location in PIXEL coordinates, not data coordinates.
        click_pixel_x, click_pixel_y = event.x, event.y
        if click_pixel_x is None or click_pixel_y is None:
            return

        xmin_view, xmax_view = ax1.get_xlim()
        visible_peak_indices = [
            i for i, (px, py) in enumerate(peak_coords) if xmin_view <= px <= xmax_view
        ]
            
        if not visible_peak_indices:
            print("No visible peaks to remove in the current view.")
            return

        pixel_distances = []
        for i in visible_peak_indices:
            px, py = peak_coords[i]
            peak_pixel = ax1.transData.transform((px, py))
            peak_pixel_x, peak_pixel_y = peak_pixel[0], peak_pixel[1]
            dist = np.hypot(peak_pixel_x - click_pixel_x, peak_pixel_y - click_pixel_y)
            pixel_distances.append(dist)

        if not pixel_distances:
            return

        closest_visible_idx = np.argmin(pixel_distances)
        global_idx_to_remove = visible_peak_indices[closest_visible_idx]
        px_to_remove, py_to_remove = peak_coords[global_idx_to_remove]
        remove_peak(px_to_remove, py_to_remove)
        print(f"Removed peak near: ({px_to_remove:.4f}, {py_to_remove:.4f})")

    # New key handler for "z": perform Y fitting for peaks within the fit limits
    elif key == 'z':
        # Prepare simlim.txt with current display limits before fitting ---
        try:
            print("Preparing simlim.txt with current display limits.")
            current_xlim = ax1.get_xlim()
            # Use safe_write_to_file for robustness, converting limits to string lines
            lines_to_write = [f"{current_xlim[0]:.33e}\n", f"{current_xlim[1]:.33e}\n"]
            safe_write_to_file("simlim.txt", lines_to_write)
        except Exception as e:
            # If we can't prepare this critical file, we should not proceed with the fit.
            messagebox.showerror("File Error", f"Could not write to simlim.txt:\n{e}")
            return # Stop the fitting process
        
        ErrorFun = fit_peaks(filename, "fitlim.txt", peaklist_filename, MIN_AMP, NUM_STARTS, 1.0e-8)

        current_xlim = ax1.get_xlim()
        lower_lim_peaklist = current_xlim[0]
        upper_lim_peaklist = current_xlim[1]
        lines_inside_xlim = filter_file_by_range(peaklist_filename, lower_lim_peaklist, upper_lim_peaklist, "Peaklist_loc.txt")

        # --- RESIDUAL PLOTTING LOGIC ---
        calculate_contour_simulation("Peaklist_loc.txt")
        if os.path.exists("Autosim.txt"):
            try:
                sim_data = np.loadtxt("Autosim.txt")
                sim_x, sim_y = sim_data[:, 0], sim_data[:, 1]
                
                # We need to ensure the simulation and experiment are on the same X grid.
                # np.interp is the safest way to do this.
                residual_y = y - np.interp(x, sim_x, sim_y)
                
                # Now we can plot on ax3 because it was declared global at the top
                ax3.clear()
                ax3.plot(x, residual_y, color='gray', linewidth=1.0, label='Residual')
                ax3.axhline(0, color='black', linestyle=':', linewidth=0.7)
                ax3.set_ylabel("Residual")
                
                # Auto-scale the y-axis for good visibility
                res_std = np.std(residual_y)
                ax3.set_ylim(-2 * res_std, 2 * res_std)
                ax3.legend(loc='upper right')

            except Exception as e:
                print(f"Error calculating or plotting residual: {e}")
        # --- END OF RESIDUAL PLOTTING LOGIC ---

        load_existing_peaks(peaklist_filename)
        plot_line("Autosim.txt", 2, 'red')
        fig.canvas.draw_idle()

    elif key == 'e':
        if event.inaxes != ax1:
            return
    
        cursor_x = event.xdata
        if cursor_x is None:
            return
    
        if len(fit_lines) >= 2:
            for line_pair in fit_lines:
                try:
                    line_pair[0].remove()
                    line_pair[1].remove()
                except (NotImplementedError, ValueError):
                    pass
            fit_lines.clear()
            with open("fitlim.txt", "w") as f:
                f.write("")
        else:
            line1 = ax1.axvline(x=cursor_x, color='black', linestyle='-', alpha=0.7)
            line2 = ax2.axvline(x=cursor_x, color='black', linestyle='-', alpha=0.7)
            fit_lines.append((line1, line2))
        
            with open("fitlim.txt", "a") as f:
                f.write(f"{cursor_x:.6f}\n")
    
        fig.canvas.draw()

    # Handle arrow keys regardless of the current axes
    elif key in ['left', 'right', 'up', 'down', '=', '-', '[', ']', 'a', 'd']:
        # This is a good place to consolidate the redraw logic
        if key in ['left', 'right', 'a', 'd']:
            current_xlim = ax1.get_xlim()
            x_range = current_xlim[1] - current_xlim[0]
            shift = 0.2 * x_range
            new_xlim = (current_xlim[0] - shift, current_xlim[1] - shift) if key == 'left' or key == 'a' else (current_xlim[0] + shift, current_xlim[1] + shift)
            ax1.set_xlim(new_xlim)
            ax2.set_xlim(new_xlim)
        elif key in ['up', 'down']:
            current_ylim1, current_ylim2, current_ylim3 = ax1.get_ylim(), ax2.get_ylim(), ax3.get_ylim()
            shift1, shift2, shift3 = 0.2 * (current_ylim1[1] - current_ylim1[0]), 0.2 * (current_ylim2[1] - current_ylim2[0]), 0.2 * (current_ylim3[1] - current_ylim3[0])
            if key == 'up':
                ax1.set_ylim(current_ylim1[0] + shift1, current_ylim1[1] + shift1)
                ax2.set_ylim(current_ylim2[0] + shift2, current_ylim2[1] + shift2)
            #    ax3.set_ylim(current_ylim3[0] + shift3, current_ylim3[1] + shift3)
            else:
                ax1.set_ylim(current_ylim1[0] - shift1, current_ylim1[1] - shift1)
                ax2.set_ylim(current_ylim2[0] - shift2, current_ylim2[1] - shift2)
            #    ax3.set_ylim(current_ylim3[0] - shift3, current_ylim3[1] - shift3)
        elif key in ['=', '-']:
            zoom_factor = 0.8 if key == '=' else 1.25
            center = np.mean(ax1.get_xlim())
            span = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * zoom_factor
            new_xlim = (center - span/2, center + span/2) # Store new limits
            ax1.set_xlim(new_xlim)
            ax2.set_xlim(new_xlim)
        elif key in ['[', ']']:
            # This helper function calculates the new zoomed limits
            def zoom_limits(lim, factor):
                center = np.mean(lim)
                span = (lim[1] - lim[0]) * factor
                return (center - span/2, center + span/2)

            # Determine the zoom factor based on the key pressed
            zoom_factor = 0.8 if key == ']' else 1.25  # ] zooms in (smaller range), [ zooms out

            # --- NEW CONTEXT-SENSITIVE LOGIC ---
            # Check if the key was pressed while the cursor was over the residuals plot
            if event.inaxes == ax3:
                # SPECIAL CASE: Zoom ONLY the y-axis of the residuals plot
                print("Zooming Residuals Y-Axis only.")
                current_ylim = ax3.get_ylim()
                ax3.set_ylim(zoom_limits(current_ylim, zoom_factor))

            else:
                # GENERAL CASE: The cursor is over ax1, ax2, or outside the plots.
                # Perform the original behavior of zooming all y-axes together.
                ylim1 = ax1.get_ylim()
                ylim2 = ax2.get_ylim()

                ax1.set_ylim(zoom_limits(ylim1, zoom_factor))
                ax2.set_ylim(zoom_limits(ylim2, zoom_factor))

        plt.draw()
        update_needed = True

    fig.canvas.draw_idle()
    if update_needed:
        _update_plots_for_view()

def on_scroll(event):
    """
    Handles mouse scroll events for Y-axis zooming.
    """

    # Ensure the scroll event happened within one of our plot areas.
    if event.inaxes is None:
        return

    # Determine the zoom factor: 'up' zooms in, 'down' zooms out.
    zoom_factor = 0.8 if event.button == 'up' else 1.25
    
    # Get the current Y-limits and the cursor's Y position.
    current_ylim = event.inaxes.get_ylim()
    y_cursor = event.ydata

    # Calculate the new Y-limits, keeping the cursor's position fixed.
    new_span = (current_ylim[1] - current_ylim[0]) * zoom_factor
    rel_pos = (y_cursor - current_ylim[0]) / (current_ylim[1] - current_ylim[0])
    new_ymin = y_cursor - new_span * rel_pos
    new_ymax = y_cursor + new_span * (1 - rel_pos)
    
    # --- Apply the new limits with context-sensitivity ---
    if event.inaxes == ax3:
        # If scrolling over the residual plot, only zoom its Y-axis.
        ax3.set_ylim(new_ymin, new_ymax)
    else:
        # Otherwise, zoom the main plot (ax1) and the secondary plot (ax2) together.
        # We need to recalculate the zoom for each axis if they have different limits.
        
        # For ax1
        if event.inaxes == ax1:
            ax1_ylim = ax1.get_ylim()
            ax1_new_span = (ax1_ylim[1] - ax1_ylim[0]) * zoom_factor
            # We assume the relative cursor position is the same for both plots
            ax1_new_ymin = y_cursor - ax1_new_span * rel_pos
            ax1_new_ymax = y_cursor + ax1_new_span * (1 - rel_pos)
            ax1.set_ylim(ax1_new_ymin, ax1_new_ymax)
            ax2.set_ylim(ax1_new_ymin, ax1_new_ymax)

        # For ax2
        else:
            ax2_ylim = ax2.get_ylim()
            ax2_new_span = (ax2_ylim[1] - ax2_ylim[0]) * zoom_factor
            # We need the cursor's y-position in ax2's data coordinates
            ax2_new_ymin = y_cursor - ax2_new_span * rel_pos
            ax2_new_ymax = y_cursor + ax2_new_span * (1 - rel_pos)
            ax2.set_ylim(ax2_new_ymin, ax2_new_ymax)
            ax1.set_ylim(ax2_new_ymin, ax2_new_ymax)

    # Redraw the figure to show the changes.
    fig.canvas.draw_idle()

def on_hover(event):
    """
    Displays a prominent, horizontal annotation and a vertical guide line when
    the mouse hovers over a peak marker or related element. The vertical line
    is shown for ALL peaks, while the annotation is only shown for assigned peaks.
    """
    global current_hover_annotation, current_hover_line, peak_plot_object, \
           base_annotation_artists, calc_base_annotation_artists, calc_peak_artists, \
           peak_data_store, peak_coords

    # --- 1. CLEANUP PHASE ---
    # This section cleans up BOTH the old annotation and the old line.
    
    # Check for and remove the previous text annotation
    if current_hover_annotation:
        # This check prevents flickering when moving the mouse over an existing annotation
        try:
            if current_hover_annotation.artist.get_window_extent().contains(event.x, event.y):
                return
        except: pass
        try: current_hover_annotation.remove()
        except: pass
        current_hover_annotation = None
        fig.canvas.draw_idle()

    # Check for and remove the previous vertical line
    if current_hover_line:
        try:
            current_hover_line.remove()
        except Exception:
            pass
        current_hover_line = None
        # The draw_idle() call for the annotation removal will handle this as well.

    if event.inaxes != ax1:
        return

    # --- 2. DETECTION PHASE ---
    # This logic finds which peak (if any) is being hovered over.
    hover_info = None
    info_source = None

    # Check for hover on a USER PEAK base annotation
    for artist in base_annotation_artists:
        if artist.contains(event)[0]:
            hover_info = {'peak_index': artist.peak_index}
            info_source = 'user'
            break

    # If not found, check for hover on a CALCULATED PEAK base annotation
    if not hover_info:
        for artist in calc_base_annotation_artists:
            if artist.contains(event)[0]:
                hover_info = artist.peak_data
                info_source = 'calc'
                break
    
    # Check for hover on a CALCULATED PEAK v-line
    if not hover_info:
        for artist in calc_peak_artists:
            if isinstance(artist, matplotlib.collections.LineCollection) and hasattr(artist, 'peak_data'):
                contains, _ = artist.contains(event)
                if contains:
                    hover_info = artist.peak_data
                    info_source = 'calc'
                    break

    # If not found, check for hover on a USER PEAK marker (annotated or not)
    if not hover_info and peak_plot_object:
        contains, ind = peak_plot_object.contains(event)
        if contains:
            try:
                xmin_view, xmax_view = ax1.get_xlim()
                visible_indices = [i for i, (px,py) in enumerate(peak_coords) if xmin_view <= px <= xmax_view]
                hover_info = {'peak_index': visible_indices[ind['ind'][0]]}
                info_source = 'user'
            except IndexError:
                return

    # --- 3. CREATION PHASE ---
    # This section now draws the line first, then conditionally adds the annotation.
    if hover_info:
        peak_x, peak_y = None, None
        assignment_str = []
        box_color = 'lightyellow'

        # STEP 3.1: Get the coordinates and assignment info based on the source
        if info_source == 'user':
            idx = hover_info['peak_index']
            peak_x, peak_y = peak_coords[idx]
            
            peak_elements = peak_data_store[idx]
            contour_type = peak_elements[3]
            assignment_start_index = 7 if contour_type == 'Voigt' else 6
            
            if len(peak_elements) > assignment_start_index:
                assignment_str = peak_elements[assignment_start_index:]
            else:
                assignment_str = []
            
        elif info_source == 'calc':
            peak_x, peak_y = hover_info['x'], hover_info['y']
            assignment_str = [hover_info['text']]
            box_color = 'lightcyan'

        # STEP 3.2: Draw the vertical line (this now happens for ALL hovered peaks)
        if peak_x is not None:
            current_hover_line = ax1.vlines(
                peak_x,
                ymin=0.0,
                ymax=peak_y,
                colors='purple',
                linestyles='--',
                linewidth=1.5,
                zorder=2
            )

        # STEP 3.3: Conditionally create and draw the text annotation ONLY if assignments exist
        if assignment_str:
            if info_source == 'user':
                annotation_lines = [" ".join(chunk[0:STATE_INDEX]) + " → " + " ".join(chunk[STATE_INDEX:ASSIGNMENT_BLOCK_SIZE])
                                    for chunk in (assignment_str[k:k+ASSIGNMENT_BLOCK_SIZE] 
                                                  for k in range(0, len(assignment_str), ASSIGNMENT_BLOCK_SIZE))
                                    if len(chunk) == ASSIGNMENT_BLOCK_SIZE]
                if not annotation_lines: return # Exit if assignments are present but malformed
                annotation_text = "\n".join(annotation_lines)
            else: # 'calc' source
                annotation_text = assignment_str[0]

            offsetbox = TextArea(annotation_text, textprops=dict(color="black", size=9))
            current_hover_annotation = AnnotationBbox(
                offsetbox, 
                (peak_x, peak_y),
                xybox=(0., 20.),
                xycoords='data',
                boxcoords="offset points",
                frameon=True,
                pad=0.4,
                arrowprops=dict(arrowstyle="-", color='gray'),
                bboxprops=dict(boxstyle="round,pad=0.4", fc=box_color, ec="black", lw=1, alpha=0.85)
            )
            
            current_hover_annotation.set_zorder(20)
            ax1.add_artist(current_hover_annotation)
        
        # Redraw the canvas to show all new artists (line and/or annotation)
        fig.canvas.draw_idle()

def on_calc_pick(event):
    """
    Handles a left-click event on a pickable calculated transition line (green v-lines).
    Highlights the selected line and writes its quantum numbers to LabelsTemp.txt.
    """
    global current_highlighted_calc_line, calc_peak_artists, ax1, fig

    # This handler should only respond to left-clicks.
    if event.mouseevent.button != MouseButton.LEFT:
        return

    # Check if the clicked artist is one of our calculated v-lines.
    # This ensures this logic doesn't fire for other pickable artists on the plot.
    if event.artist in calc_peak_artists and hasattr(event.artist, 'peak_data'):
        
        # --- 1. Visual Highlighting Logic ---
        # Remove the previous highlight if it exists.
        if current_highlighted_calc_line:
            try:
                current_highlighted_calc_line.remove()
            except Exception:
                pass # Ignore if it's already gone
        
        # Get data from the clicked artist.
        peak_data = event.artist.peak_data
        x_val = peak_data['x']
        y_val = peak_data['y']
        
        # Draw a new, more prominent line as the highlight.
        current_highlighted_calc_line = ax1.vlines(
            x_val, ymin=0, ymax=y_val,
            colors='lime',      # A brighter green color
            linewidths=2.5,     # A thicker line
            alpha=0.9,
            zorder=15           # Ensure it's drawn on top of other elements
        )

        # --- 2. Data Handling Logic ---
        annotation_text = peak_data['text']
        
        try:
            # Parse the "1 2 3 -> 4 5 6" string back into number tuples
            parts = annotation_text.split(' → ')
            lower_qns_str_list = parts[0].split()
            upper_qns_str_list = parts[1].split()

            # Convert string lists to tuples of integers
            lower_qns_tuple = tuple(map(int, lower_qns_str_list))
            upper_qns_tuple = tuple(map(int, upper_qns_str_list))
            
            # Use the existing helper function to write to the file
            _write_selection_to_labels_temp(lower_qns_tuple, upper_qns_tuple)
            print(f"Copied calculated transition to LabelsTemp.txt: {annotation_text}")

        except (IndexError, ValueError) as e:
            print(f"Error parsing QN string from calculated line: '{annotation_text}'. Error: {e}")

        # --- 3. Redraw the Canvas ---
        fig.canvas.draw_idle()

def handle_pick(event):
    """Handle double-click on peak points to open the editor.
    This version correctly calculates distance in pixel space to select the visually closest peak.
    """
    global input_window_open, peak_coords, peak_data_store, ax1, fig
    global STATE_INDEX, ASSIGNMENT_BLOCK_SIZE, DEFAULT_FWHM, DEFAULT_FWHM_G, DEFAULT_FWHM_L

    if event.inaxes != ax1 or input_window_open:
        return

    # Get the click location in PIXEL coordinates
    click_pixel_x, click_pixel_y = event.x, event.y
    if click_pixel_x is None or click_pixel_y is None:
        return # Click was outside the figure area

    # --- Find the visually closest peak ---

    # 1. Filter for peaks visible in the current view to improve performance
    #    and prevent selecting off-screen peaks.
    xmin_view, xmax_view = ax1.get_xlim()
    visible_peak_indices = [
        i for i, (px, py) in enumerate(peak_coords) if xmin_view <= px <= xmax_view
    ]
    
    if not visible_peak_indices:
        return # No visible peaks to select

    pixel_distances = []
    # 2. Loop through only the VISIBLE peaks
    for i in visible_peak_indices:
        px, py = peak_coords[i]
        
        # 3. Transform the peak's data coordinates to pixel coordinates
        peak_pixel = ax1.transData.transform((px, py))
        peak_pixel_x, peak_pixel_y = peak_pixel[0], peak_pixel[1]
        
        # 4. Calculate distance in pixels
        dist = np.hypot(peak_pixel_x - click_pixel_x, peak_pixel_y - click_pixel_y)
        pixel_distances.append(dist)

    if not pixel_distances:
        return

    # 5. Find the index of the minimum distance in our filtered list
    closest_visible_idx = np.argmin(pixel_distances)
    
    # 6. Map this back to the original index in the global peak_coords list
    idx = visible_peak_indices[closest_visible_idx]
    
    # --- The rest of the function continues as before, now with the correct 'idx' ---
    peak_x_val, peak_y_val = peak_coords[idx]

    # Read the existing line from Peaklist to get 'parts'
    try:
        with open(peaklist_filename, "r") as f:
            lines = f.readlines()
            if idx >= len(lines):
                messagebox.showerror("Error", f"Internal error: Could not find peak data for index {idx}.")
                return
            line = lines[idx].strip()
            parts = line.split()
    except Exception as e:
        messagebox.showerror("Error", f"Error reading peak data: {e}")
        return

    # --- Create the Main Dialog Window ---
    input_window_open = True
    fig_manager = plt.get_current_fig_manager()
    root = fig_manager.window
    
    # Capture cursor's global screen coordinates ---
    # We ask the main window ('root') for the pointer's current position.
    cursor_x_screen = root.winfo_pointerx()
    cursor_y_screen = root.winfo_pointery()
    
    dialog = Toplevel(root)
    dialog.title(f"Edit Peak & Assign QNs (Peak at {peak_x_val:.6f})")

    # Use the helper function to position and size the window
    position_dialog_near_cursor(dialog, width=500, height=600)

    # Frame for Peak Parameters ---
    params_frame = Frame(dialog, bd=2, relief="groove")
    params_frame.pack(padx=10, pady=10, fill='x')
    params_frame.grid_columnconfigure(1, weight=1)
    params_frame.grid_columnconfigure(3, weight=1)

    Label(params_frame, text="Peak Parameters").grid(row=0, column=0, columnspan=4, pady=(2, 5))

    # Center Entry
    Label(params_frame, text="Center:").grid(row=1, column=0, sticky='w', padx=5)
    center_entry = Entry(params_frame, width=20)
    center_entry.grid(row=1, column=1, sticky='ew', padx=5)
    center_entry.insert(0, parts[1])

    # Amplitude Entry
    Label(params_frame, text="Amplitude:").grid(row=1, column=2, sticky='w', padx=5)
    amplitude_entry = Entry(params_frame, width=20)
    amplitude_entry.grid(row=1, column=3, sticky='ew', padx=5)
    amplitude_entry.insert(0, parts[2])

    # Contour Type Dropdown
    Label(params_frame, text="Contour:").grid(row=2, column=0, sticky='w', padx=5)
    contour_var = tk.StringVar(dialog)
    contour_options = ['Gaussian', 'Lorentzian', 'Voigt']
    contour_menu = tk.OptionMenu(params_frame, contour_var, *contour_options) # The command is set later
    contour_menu.grid(row=2, column=1, sticky='ew', padx=5, pady=5)

    # Fix Peak Parameters Checkbox ---
    mark_peak_var = tk.BooleanVar(dialog)
    mark_peak_checkbox = tk.Checkbutton(params_frame, text="Fix parameters", variable=mark_peak_var)
    mark_peak_checkbox.grid(row=3, column=0, columnspan=2, sticky='w', padx=5, pady=5)
    
    # Set the initial state of the checkbox based on the symbol from the file
    if parts[0] == 'M':
        mark_peak_var.set(True)
    else:
        mark_peak_var.set(False)

    # Dynamic FWHM Entry Fields ---
    fwhm_g_label = Label(params_frame, text="FWHM (G):")
    fwhm_g_entry = Entry(params_frame, width=20)
    fwhm_l_label = Label(params_frame, text="FWHM (L):")
    fwhm_l_entry = Entry(params_frame, width=20)
    fwhm_label = Label(params_frame, text="FWHM:")
    fwhm_entry = Entry(params_frame, width=20)

    # This function now also populates the fields with the global defaults.
    def _update_fwhm_fields(*args):
        """Show/hide FWHM entry fields AND POPULATE WITH DEFAULTS when contour is changed."""
        selected_contour = contour_var.get()

        # Hide all first
        fwhm_label.grid_forget(); fwhm_entry.grid_forget()
        fwhm_g_label.grid_forget(); fwhm_g_entry.grid_forget()
        fwhm_l_label.grid_forget(); fwhm_l_entry.grid_forget()

        if selected_contour == 'Voigt':
            # Show Voigt-specific widgets
            fwhm_g_label.grid(row=2, column=2, sticky='w', padx=5)
            fwhm_g_entry.grid(row=2, column=3, sticky='ew', padx=5, pady=5)
            fwhm_l_label.grid(row=3, column=2, sticky='w', padx=5)
            fwhm_l_entry.grid(row=3, column=3, sticky='ew', padx=5, pady=5)
            
            # Clear and populate with Voigt defaults
            fwhm_g_entry.delete(0, END)
            fwhm_g_entry.insert(0, str(DEFAULT_FWHM_G))
            fwhm_l_entry.delete(0, END)
            fwhm_l_entry.insert(0, str(DEFAULT_FWHM_L))
        else: # Gaussian or Lorentzian
            # Show the generic FWHM widget
            fwhm_label.grid(row=2, column=2, sticky='w', padx=5)
            fwhm_entry.grid(row=2, column=3, sticky='ew', padx=5, pady=5)

            # Clear and populate with the generic default
            fwhm_entry.delete(0, END)
            fwhm_entry.insert(0, str(DEFAULT_FWHM))

    # 1. Get the initial contour type from the file
    contour_type = parts[3]
    contour_var.set(contour_type if contour_type in contour_options else contour_options[0])

    # 2. Populate FWHM fields with values from the file *before* attaching the trace
    if contour_type == 'Voigt':
        _update_fwhm_fields() # Make sure the correct widgets are visible
        fwhm_g_entry.delete(0, END)
        fwhm_l_entry.delete(0, END)
        if len(parts) > 4:
            fwhm_g_entry.insert(0, parts[4])
            fwhm_l_entry.insert(0, parts[5])
        else: # Fallback for corrupted/incomplete line
            fwhm_g_entry.insert(0, str(DEFAULT_FWHM_G))
            fwhm_l_entry.insert(0, str(DEFAULT_FWHM_L))
        assignment_start_index = 7
    else: # Gaussian or Lorentzian
        _update_fwhm_fields() # Make sure the correct widget is visible
        fwhm_entry.delete(0, END)
        fwhm_entry.insert(0, parts[4])
        assignment_start_index = 6

    # 3. NOW, attach the trace. Any subsequent change by the user will trigger
    #    the `_update_fwhm_fields` function and populate with defaults.
    contour_var.trace_add("write", _update_fwhm_fields)
    
    # --- [The rest of the handle_pick function remains unchanged] ---

    # --- Check for LabelsTemp.txt (suggestion section) ---
    labels_temp_file = "LabelsTemp.txt"
    labels_temp_content = None
    if os.path.exists(labels_temp_file) and os.path.getsize(labels_temp_file) > 0:
        with open(labels_temp_file, 'r') as f_temp:
            labels_temp_content = f_temp.read().strip()
        
        temp_label_frame = Frame(dialog)
        temp_label_frame.pack(padx=10, pady=(0, 5), fill='x')
        Label(temp_label_frame, text="Suggestion:", font=('Arial', 9, 'italic')).pack(side=LEFT)
        Label(temp_label_frame, text=labels_temp_content, font=("Courier New", 9)).pack(side=LEFT, padx=5)
        
        def add_temp_labels_command():
            if labels_temp_content:
                text.insert("1.0", labels_temp_content + '\n')
        
        Button(temp_label_frame, text="Add", command=add_temp_labels_command).pack(side=LEFT, padx=5)

    # --- Main Text Input Section for Quantum Numbers ---
    qn_frame = Frame(dialog)
    qn_frame.pack(padx=10, pady=5, fill=BOTH, expand=True)

    Label(qn_frame, text=f"Quantum Number Assignments ({ASSIGNMENT_BLOCK_SIZE} integers per line):").pack(anchor='w')
    
    scroll = Scrollbar(qn_frame)
    text = Text(qn_frame, height=10, width=50, yscrollcommand=scroll.set, wrap='word')
    scroll.config(command=text.yview)
    scroll.pack(side=RIGHT, fill=Y)
    text.pack(side=LEFT, fill=BOTH, expand=True)

    if len(parts) > assignment_start_index:
        existing_numbers_str = parts[assignment_start_index:]
    else:
        existing_numbers_str = []
        
    if existing_numbers_str:
        chunks = [existing_numbers_str[i:i+ASSIGNMENT_BLOCK_SIZE] for i in range(0, len(existing_numbers_str), ASSIGNMENT_BLOCK_SIZE)]
        initial_text = "\n".join(" ".join(chunk) for chunk in chunks)
        if initial_text:
            text.insert(END, initial_text)
            if not initial_text.endswith('\n'):
                text.insert(END, "\n")

    # --- Apply Changes Function ---
    def apply_changes():
        global peak_data_store, peak_coords
        
        # --- 1. Get and Validate new Peak Parameters ---
        try:
            new_center = float(center_entry.get())
            new_amp = float(amplitude_entry.get())
            new_contour = contour_var.get()
            fwhm_parts = []
            if new_contour == 'Voigt':
                try:
                    new_fwhm_g = float(fwhm_g_entry.get())
                    new_fwhm_l = float(fwhm_l_entry.get())
                    fwhm_parts = [f"{new_fwhm_g:.4e}", f"{new_fwhm_l:.4e}"]
                    
                    sigma = new_fwhm_g / (2 * np.sqrt(2 * np.log(2)))
                    gamma = new_fwhm_l / 2.0
                    scaling_factor = voigt_profile(0, sigma, gamma)
                    new_integral = new_amp / scaling_factor if scaling_factor > 1e-10 else 0.0

                except ValueError:
                    messagebox.showerror("Input Error", "Voigt FWHM (G) and FWHM (L) must be valid numbers.", parent=dialog)
                    return
            else:
                try:
                    new_fwhm = float(fwhm_entry.get())
                    fwhm_parts = [f"{new_fwhm:.4e}"]
                    
                    if new_contour == 'Lorentzian':
                        new_integral = new_amp * new_fwhm * (np.pi / 2.0)
                    else: # Gaussian
                        new_integral = new_amp * new_fwhm * np.sqrt(np.pi) / (2 * np.sqrt(np.log(2)))

                except ValueError:
                    messagebox.showerror("Input Error", "FWHM must be a valid number.", parent=dialog)
                    return
        except ValueError:
            messagebox.showerror("Input Error", "Center, Amplitude, and FWHM must be valid numbers.", parent=dialog)
            return
        
        # --- 3. Get new Quantum Number Assignments ---
        input_text = text.get("1.0", END).strip()
        new_assignment_numbers_str = []
        if input_text:
            for line_num, line in enumerate(input_text.split('\n')):
                line = line.strip()
                if not line: continue
                numbers_in_line = line.split()
                if len(numbers_in_line) != ASSIGNMENT_BLOCK_SIZE:
                    messagebox.showwarning("Input Error", f"Invalid QN on line {line_num+1}. Requires {ASSIGNMENT_BLOCK_SIZE} integers.", parent=dialog)
                    continue
                try:
                    new_assignment_numbers_str.extend([str(int(n)) for n in numbers_in_line])
                except ValueError:
                    messagebox.showwarning("Input Error", f"Non-integer value found in QN line {line_num+1}.", parent=dialog)
                    continue
        
        symbol_to_save = 'M' if mark_peak_var.get() else 'S'
        # --- 4. Construct the new line for Peaklist ---
        new_base_parts = [
            symbol_to_save,
            f"{new_center:.6f}",
            f"{new_amp:.6e}",
            new_contour,
        ] + fwhm_parts + [f"{new_integral:.6e}"]
        
        new_line_content_parts = new_base_parts + new_assignment_numbers_str
        new_line_content_str = " ".join(new_line_content_parts) + "\n"

        # --- 5. Save to File and Update In-Memory Stores ---
        try:
            with open(peaklist_filename, "r") as f_read:
                lines = f_read.readlines()
            
            lines[idx] = new_line_content_str
            
            with open(peaklist_filename, "w") as f_write:
                f_write.writelines(lines)
            
            print(f"Updated peak {idx} in Peaklist")

            peak_data_store[idx] = new_line_content_parts
            peak_coords[idx] = (new_center, new_amp)
            
            load_existing_peaks(peaklist_filename)
            if fig and fig.canvas:
                fig.canvas.draw_idle()
            
        except Exception as e:
            messagebox.showerror("File Error", f"Failed to save changes to Peaklist:\n{str(e)}", parent=dialog)

    # --- Final Buttons and Window Management ---
    button_frame = Frame(dialog)
    button_frame.pack(pady=10, padx=10, fill='x')

    def clear_text_area():
        text.delete("1.0", END)

    def apply_and_close():
        apply_changes()
        if dialog.winfo_exists():
            dialog.destroy()
            globals()['input_window_open'] = False

    def cancel_and_close():
        dialog.destroy()
        globals()['input_window_open'] = False

    Button(button_frame, text="Apply & Close", command=apply_and_close).pack(side=LEFT, padx=10)
    Button(button_frame, text="Clear QNs", command=clear_text_area).pack(side=LEFT, padx=5)
    Button(button_frame, text="Cancel", command=cancel_and_close).pack(side=RIGHT, padx=10)

    dialog.protocol("WM_DELETE_WINDOW", cancel_and_close)
    dialog.lift()
    dialog.focus_force()

def plot_line(filename, graph_num, color): # <-- Add color parameter with default
    """
    Plots data from a two-column file as a line on the specified graph
    using the specified color.
    Removes the previous line plotted by this function on the same graph.

    Parameters:
        filename (str): Path to the file containing X and Y data (two columns).
        graph_num (int): Target graph number (1 for upper plot ax1, 2 for lower plot ax2).
        color (str): Color specification for the line (e.g., 'red', 'blue', '#FF5733').
                     Defaults to 'black'.
    """
    global line_plot_artists, ax1, ax2, fig # Need access to globals

    # --- Validate Inputs and Environment ---
    target_ax = None
    if graph_num == 1:
        if 'ax1' in globals() and ax1 is not None:
            target_ax = ax1
        else:
            print(f"Error (plot_line): Upper graph (ax1) not available.")
            return
    elif graph_num == 2:
        if 'ax2' in globals() and ax2 is not None:
            target_ax = ax2
        else:
            print(f"Error (plot_line): Lower graph (ax2) not available.")
            return
    else:
        print(f"Error (plot_line): Invalid graph_num ({graph_num}). Use 1 or 2.")
        return

    if not fig or not fig.canvas:
         print(f"Error (plot_line): Figure canvas not available.")
         return

    # --- Remove Previous Plot by this function on this axis ---
    previous_artist = line_plot_artists.get(graph_num, None)
    if previous_artist is not None:
        try:
            if previous_artist in target_ax.lines:
                previous_artist.remove()
            # else: pass # Already removed
        except Exception as e:
            print(f"Warning (plot_line): Could not remove previous line artist for ax{graph_num}: {e}")
        finally:
             line_plot_artists[graph_num] = None # Always clear reference


    # --- Read Data ---
    try:
        if not os.path.exists(filename):
            print(f"Error (plot_line): File not found: {filename}")
            line_plot_artists[graph_num] = None
            return

        data = np.loadtxt(filename)

        if data.size == 0:
            print(f"Warning (plot_line): File is empty: {filename}")
            line_plot_artists[graph_num] = None
            if fig and fig.canvas: fig.canvas.draw_idle() # Redraw to show removal
            return
        if data.ndim != 2 or data.shape[1] != 2:
            print(f"Error (plot_line): File format incorrect in {filename}. Expected 2 columns.")
            line_plot_artists[graph_num] = None
            return

        x_data = data[:, 0]
        y_data = data[:, 1]

    except Exception as e:
        print(f"Error (plot_line): Failed to read or process file {filename}: {e}")
        line_plot_artists[graph_num] = None
        return

    # --- Plot New Data ---
    try:
        # Use the 'color' argument passed to the function
        line_artist_list = target_ax.plot(x_data, y_data, '-',
                                         color=color, # <-- Use the color argument here
                                         linewidth=1,
                                         label=f'Line from {os.path.basename(filename)}')
        if line_artist_list:
             new_artist = line_artist_list[0]
             line_plot_artists[graph_num] = new_artist
             print(f"Successfully plotted {filename} on graph {graph_num} with color '{color}'")
             # Optional: target_ax.legend()
        else:
             print(f"Warning (plot_line): Plotting {filename} did not return a valid artist.")
             line_plot_artists[graph_num] = None

    except Exception as e:
        print(f"Error (plot_line): Failed to plot data from {filename} on graph {graph_num}: {e}")
        line_plot_artists[graph_num] = None

    # --- Redraw Canvas ---
    try:
        fig.canvas.draw_idle()
    except Exception as e:
        print(f"Warning (plot_line): Failed to redraw canvas: {e}")

def show_spectrum_processing_dialog():
    """
    Creates and displays a modular dialog for various spectrum processing tasks.
    The UI dynamically changes based on the selected task from a dropdown menu.
    """
    global spectrum_processing_window_open, filename, x, y

    if spectrum_processing_window_open:
        return
    spectrum_processing_window_open = True

    # --- Window and Frame Setup ---
    fig_manager = plt.get_current_fig_manager()
    root = fig_manager.window if fig_manager and hasattr(fig_manager, 'window') else tk.Tk()

    dialog = Toplevel(root)
    dialog.title("Spectrum Processing")
    position_dialog_near_cursor(dialog, width=800, height=700) # Increased height for new elements

    # --- Main Layout Frames ---
    plot_frame = Frame(dialog)
    plot_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
    
    mode_selection_frame = Frame(dialog)
    mode_selection_frame.pack(fill='x', padx=10, pady=5)

    # This container will hold the UI specific to the selected task
    task_frame_container = Frame(dialog)
    task_frame_container.pack(fill='x', padx=10, pady=5)
    
    dialog_button_frame = Frame(dialog)
    dialog_button_frame.pack(pady=(5, 10), padx=10, fill='x')

    # --- Embed Matplotlib Plot ---
    dialog.fig, dialog.ax = plt.subplots(figsize=(6, 4))
    dialog.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
    
    dialog.ax.plot(x, y, 'b-', label='Original Spectrum', linewidth=1)
    dialog.ax.set_title("Spectrum Processing Preview")
    dialog.ax.set_xlabel("X-axis")
    dialog.ax.set_ylabel("Y-axis")
    dialog.ax.grid(True, linestyle=':', alpha=0.6)

    dialog.canvas = FigureCanvasTkAgg(dialog.fig, master=plot_frame)
    canvas_widget = dialog.canvas.get_tk_widget()
    canvas_widget.pack(fill=BOTH, expand=True)
    
    toolbar = NavigationToolbar2Tk(dialog.canvas, plot_frame)
    toolbar.update()
    
    # Store plot artists on the dialog object to manage them from helper functions
    dialog.corrected_line_artist = None
    dialog.baseline_line_artist = None
    dialog.band_search_baseline_1 = None
    dialog.band_search_baseline_2 = None
    
    dialog.ax.legend()
    dialog.canvas.draw()

    # =====================================================================
    # --- TASK UI BUILDER: Baseline Correction ---
    # =====================================================================
    def _build_baseline_correction_ui(parent_frame):
        """Creates and packs all UI widgets for the baseline correction task."""
        
        def _run_correction_and_plot():
            """The specific action for the 'Perform Correction' button."""
            try:
                win_size = int(window_size_entry.get())
                savgol_win = int(savgol_window_entry.get())
                savgol_poly = int(savgol_poly_order_entry.get())
                mode = mode_var.get()
                
                corrected_y, baseline_y = baseline_correction(x, y, win_size, savgol_win, savgol_poly, mode)

                if dialog.corrected_line_artist: dialog.corrected_line_artist.remove()
                if dialog.baseline_line_artist: dialog.baseline_line_artist.remove()

                dialog.corrected_line_artist, = dialog.ax.plot(x, corrected_y, 'g-', label='Corrected', linewidth=1.2)
                dialog.baseline_line_artist, = dialog.ax.plot(x, baseline_y, 'r--', label='Smoothed Baseline', linewidth=1)
                
                dialog.ax.legend()
                dialog.canvas.draw_idle()

                output_filename = filedialog.asksaveasfilename(
                    title="Save Corrected Spectrum As", defaultextension=".txt",
                    filetypes=[("Text files", "*.txt"), ("All files", "*.*")], parent=dialog)
                
                if output_filename:
                    np.savetxt(output_filename, np.column_stack((x, corrected_y)), fmt='%.6f', delimiter=' ')
                    messagebox.showinfo("Success", f"Corrected data saved to:\n{output_filename}", parent=dialog)
            except ValueError as e:
                messagebox.showerror("Invalid Input", f"Please check your parameters.\n\nError: {e}", parent=dialog)
            except Exception as e:
                messagebox.showerror("Correction Failed", f"An error occurred during correction:\n\n{e}", parent=dialog)

        # --- Create Widgets ---
        params_frame = Frame(parent_frame, bd=2, relief="groove")
        params_frame.pack(fill='x', expand=False)
        
        action_button_frame = Frame(parent_frame)
        action_button_frame.pack(pady=(5, 0), fill='x')

        Label(params_frame, text="Step 1: Baseline Envelope (Rolling Ball)", font=('Arial', 10, 'bold')).pack(pady=(5,2))
        win_frame = Frame(params_frame); win_frame.pack(fill='x', padx=10, pady=3)
        Label(win_frame, text="Window Size:", width=20, anchor='w').pack(side=LEFT)
        window_size_entry = Entry(win_frame); window_size_entry.insert(0, "400"); window_size_entry.pack(side=LEFT, fill='x', expand=True)

        Label(params_frame, text="Step 2: Baseline Smoothing (Savitzky-Golay)", font=('Arial', 10, 'bold')).pack(pady=(10,2))
        savgol_win_frame = Frame(params_frame); savgol_win_frame.pack(fill='x', padx=10, pady=3)
        Label(savgol_win_frame, text="Savgol Window Length:", width=20, anchor='w').pack(side=LEFT)
        savgol_window_entry = Entry(savgol_win_frame); savgol_window_entry.insert(0, "21"); savgol_window_entry.pack(side=LEFT, fill='x', expand=True)

        savgol_poly_frame = Frame(params_frame); savgol_poly_frame.pack(fill='x', padx=10, pady=3)
        Label(savgol_poly_frame, text="Savgol Poly Order:", width=20, anchor='w').pack(side=LEFT)
        savgol_poly_order_entry = Entry(savgol_poly_frame); savgol_poly_order_entry.insert(0, "3"); savgol_poly_order_entry.pack(side=LEFT, fill='x', expand=True)

        mode_frame = Frame(params_frame); mode_frame.pack(fill='x', padx=10, pady=(10, 5))
        Label(mode_frame, text="Edge Mode (for both steps):", width=20, anchor='w').pack(side=LEFT)
        mode_var = tk.StringVar(dialog); modes = ['reflect', 'constant', 'nearest', 'mirror', 'wrap']; mode_var.set(modes[3])
        OptionMenu(mode_frame, mode_var, *modes).pack(side=LEFT, fill='x', expand=True)
        
        Button(action_button_frame, text="Perform Correction", command=_run_correction_and_plot).pack(side=LEFT, padx=5)

    # =====================================================================
    # --- TASK UI BUILDER: Band Centers Search ---
    # =====================================================================
    def _build_band_search_ui(parent_frame):
        """Creates and packs all UI widgets for the band centers search task."""
        
        # This frame will hold the UI for the selected method
        method_ui_frame = Frame(parent_frame)

        def _build_baseline_density_ui(method_parent_frame):
            """Creates the specific UI for the 'baseline density' method."""
            
            def _calculate_and_plot_baselines():
                """Action for the 'Calculate Baselines' button."""
                try:
                    win1 = int(win1_entry.get())
                    win2 = int(win2_entry.get())
                    if win1 <= 0 or win2 <= 0: raise ValueError("Window sizes must be positive.")
                except ValueError as e:
                    messagebox.showerror("Invalid Input", f"Please enter valid positive integers for window sizes.\n\nError: {e}", parent=dialog)
                    return
                
                # --- Remove previous baseline plots if they exist ---
                if dialog.band_search_baseline_1: dialog.band_search_baseline_1.remove()
                if dialog.band_search_baseline_2: dialog.band_search_baseline_2.remove()

                # --- Calculate and plot new baselines ---
                baseline1 = minimum_filter(y, size=win1, mode='mirror')
                baseline2 = minimum_filter(y, size=win2, mode='mirror')
                
                dialog.band_search_baseline_1, = dialog.ax.plot(x, baseline1, 'r-', label=f'Baseline (Win={win1})')
                dialog.band_search_baseline_2, = dialog.ax.plot(x, baseline2, 'g-', label=f'Baseline (Win={win2})')

                dialog.ax.legend()
                dialog.canvas.draw_idle()

            # --- Create Widgets for this method ---
            params_frame = Frame(method_parent_frame, bd=2, relief="groove")
            params_frame.pack(fill='x', expand=False, pady=(10, 0))
            
            action_button_frame = Frame(method_parent_frame)
            action_button_frame.pack(pady=(5, 0), fill='x')

            win_frame = Frame(params_frame)
            win_frame.pack(fill='x', padx=10, pady=10)
            win_frame.columnconfigure(1, weight=1)
            win_frame.columnconfigure(3, weight=1)

            Label(win_frame, text="Window 1:").grid(row=0, column=0, sticky='w')
            win1_entry = Entry(win_frame); win1_entry.insert(0, "50"); win1_entry.grid(row=0, column=1, sticky='ew', padx=(5, 10))
            
            Label(win_frame, text="Window 2:").grid(row=0, column=2, sticky='w')
            win2_entry = Entry(win_frame); win2_entry.insert(0, "400"); win2_entry.grid(row=0, column=3, sticky='ew', padx=5)
            
            Button(action_button_frame, text="Calculate Baselines", command=_calculate_and_plot_baselines).pack(side=LEFT, padx=5)

        def _on_band_search_method_change(*args):
            """Controls which method's UI is visible."""
            for widget in method_ui_frame.winfo_children():
                widget.destroy() # Clear out the old UI
            
            selected_method = band_search_method_var.get()
            if selected_method == "baseline density":
                _build_baseline_density_ui(method_ui_frame)
            # Add elif for other future methods here

        # --- Create Widgets for the main Band Search task ---
        method_selection_frame = Frame(parent_frame)
        method_selection_frame.pack(fill='x', pady=5)
        
        Label(method_selection_frame, text="Band Center Search Method:").pack(side=LEFT, padx=(0, 10))
        band_search_method_var = tk.StringVar(dialog)
        method_options = ["baseline density"] # Add more methods here in the future
        band_search_method_var.set(method_options[0])
        
        OptionMenu(method_selection_frame, band_search_method_var, *method_options, command=_on_band_search_method_change).pack(side=LEFT, fill='x', expand=True)
        
        method_ui_frame.pack(fill='x') # Pack the container for the method's specific UI
        _on_band_search_method_change() # Call once to build the initial UI

    # =====================================================================
    # --- MASTER CONTROLLER and DIALOG EXIT ---
    # =====================================================================
    def _on_processing_mode_change(*args):
        """Master controller. Destroys old task UI and builds new task UI."""
        # Clear any old plot artists from the preview
        if dialog.corrected_line_artist: dialog.corrected_line_artist.remove(); dialog.corrected_line_artist = None
        if dialog.baseline_line_artist: dialog.baseline_line_artist.remove(); dialog.baseline_line_artist = None
        if dialog.band_search_baseline_1: dialog.band_search_baseline_1.remove(); dialog.band_search_baseline_1 = None
        if dialog.band_search_baseline_2: dialog.band_search_baseline_2.remove(); dialog.band_search_baseline_2 = None
        dialog.ax.legend()
        dialog.canvas.draw_idle()

        # Destroy all widgets in the container to make room for the new UI
        for widget in task_frame_container.winfo_children():
            widget.destroy()
        
        selected_mode = processing_mode_var.get()
        if selected_mode == "Baseline correction":
            _build_baseline_correction_ui(task_frame_container)
        elif selected_mode == "Band centers search":
            _build_band_search_ui(task_frame_container)
    
    def on_close():
        globals()['spectrum_processing_window_open'] = False 
        plt.close(dialog.fig)
        dialog.destroy()

    # --- Create Top-Level Dropdown and Final Button ---
    Label(mode_selection_frame, text="Select Processing Task:").pack(side=LEFT, padx=(0, 10))
    processing_mode_var = tk.StringVar(dialog)
    processing_options = ["Baseline correction", "Band centers search"] 
    processing_mode_var.set(processing_options[0])
    OptionMenu(mode_selection_frame, processing_mode_var, *processing_options, command=_on_processing_mode_change).pack(side=LEFT, fill='x', expand=True)

    Button(dialog_button_frame, text="Close", command=on_close).pack(side=RIGHT, padx=5)
    dialog.protocol("WM_DELETE_WINDOW", on_close)

    # --- Initialize UI State ---
    _on_processing_mode_change()

    dialog.lift()
    dialog.attributes('-topmost', True)
    dialog.after_idle(dialog.attributes, '-topmost', False)
    dialog.focus_force()

def baseline_correction(
    x_data: np.ndarray,                
    y_data: np.ndarray,                
    window_size: int,
    savgol_window_length: int,
    savgol_poly_order: int,
    mode: str
) -> tuple[np.ndarray, np.ndarray]:    
    """
    Apply baseline correction and return the corrected data and the baseline.

    This two-step process first estimates the baseline using a rolling ball
    algorithm and then smooths this estimated baseline before subtraction.

    Parameters:
        x_data (np.ndarray):
            The X-values of the spectrum.
        y_data (np.ndarray):
            The Y-values of the spectrum.
        window_size (int):
            The size of the window for the initial minimum filter (rolling ball).
        savgol_window_length (int):
            The window length for smoothing the estimated baseline. Must be odd.
        savgol_poly_order (int):
            The polynomial order for the Savitzky-Golay smoothing.
        mode (str):
            The mode for handling signal boundaries (e.g., 'reflect').

    Returns:
        tuple[np.ndarray, np.ndarray]:
            A tuple containing (corrected_y, smoothed_baseline).

    Raises:
        ValueError: If window sizes or polynomial order are invalid.
    """

    # --- Validation for window sizes and parameters ---
    if window_size < 3:
        raise ValueError("Rolling ball window size must be >= 3 points")
    if window_size > len(y_data):
        raise ValueError("Rolling ball window size cannot exceed data length")
    if savgol_window_length % 2 == 0 or savgol_window_length <= 0:
        raise ValueError("Savgol window length must be a positive odd integer.")
    if savgol_poly_order >= savgol_window_length:
        raise ValueError("Savgol polynomial order must be less than the window length.")

    # Step 1: Rolling ball baseline estimation
    baseline = minimum_filter(y_data, size=window_size, mode=mode)

    # Step 2: Use Savitzky-Golay to smooth the estimated baseline
    smoothed_baseline = savgol_filter(
        baseline,
        window_length=savgol_window_length,
        polyorder=savgol_poly_order,
        mode=mode
    )

    # Step 3: Calculate corrected spectrum
    corrected_y = y_data - smoothed_baseline

    return corrected_y, smoothed_baseline

# --- Helper function to write to LabelsTemp.txt ---
def _write_selection_to_labels_temp(qns1_tuple, qns2_tuple):
    """Formats QN tuples and writes them to LabelsTemp.txt."""
    labels_temp_file = "LabelsTemp.txt"
    try:
        qns1_str = ' '.join(map(str, qns1_tuple))
        qns2_str = ' '.join(map(str, qns2_tuple))
        output_line = f"{qns1_str} {qns2_str}" # Combine lower and upper QNs
        with open(labels_temp_file, 'w') as f_temp:
            f_temp.write(output_line + '\n')
    except Exception as e:
        print(f"Error writing to {labels_temp_file}: {e}")

def Interpretation(calculated_file, lower_energies_file='Lower.txt', upper_energies_file='Upper.txt'):
    # --- Declarations ---
    import tkinter as tk # Ensure tk is imported within the function scope too

    global destination_numbers, fig, ax1 # Allow modification if needed

    # Local state for this instance of Interpretation
    entries = []
    filtered_entries = []
    current_line = None # Line drawn on the main plot (ax1)
    loomis_wood_data_store = {} # For calculated LW points
    loomis_wood_data_store_exp = {} # For experimental LW points

    # State specifically for the integrated Loomis-Wood window
    lw_window = None
    lw_fig = None
    lw_ax = None
    lw_folding_entry = None
    lw_intensity_entry = None
    lw_ka_entry = None
    lw_qn_index_entry = None # Entry for QN index to filter
    lw_canvas = None # Reference to the canvas widget
    lw_plot_exp_var = None # BooleanVar for the checkbox
    lw_highlight_marker = None

    # --- File Reading (Calculated Data) ---
    try:
        # Ensure CALC_SCALE is defined before use
        if 'CALC_SCALE' not in globals(): raise NameError("Global variable 'CALC_SCALE' is not defined.")
        if 'STATE_INDEX' not in globals(): raise NameError("Global variable 'STATE_INDEX' is not defined.")

        expected_min_cols = 2 + (2 * STATE_INDEX) # X, Y, Lower QNs, Upper QNs

        with open(calculated_file, 'r') as f:
            line_num = 0
            for line in f:
                line_num += 1
                parts = line.strip().split()
                if len(parts) < expected_min_cols:
                    if len(parts) > 0: # Don't warn for blank lines
                       print(f"Skipping line {line_num} in {calculated_file} (expected >= {expected_min_cols} parts, got {len(parts)}).")
                    continue

                # --- Indices for QNs based on STATE_INDEX ---
                lower_qn_start_idx = 2
                lower_qn_end_idx = lower_qn_start_idx + STATE_INDEX
                upper_qn_start_idx = lower_qn_end_idx
                upper_qn_end_idx = upper_qn_start_idx + STATE_INDEX
                # --- End Indices ---

                try:
                    freq = float(parts[0])
                    intensity_str = parts[1].replace('*', '') # Example handling
                    intensity = float(intensity_str) * CALC_SCALE

                    # Extract QNs using dynamic indices and convert to tuple of ints
                    lower_qns = tuple(map(int, parts[lower_qn_start_idx:lower_qn_end_idx]))
                    upper_qns = tuple(map(int, parts[upper_qn_start_idx:upper_qn_end_idx]))

                    # Add to entries list
                    entries.append({'freq': freq, 'intensity': intensity, 'triple1': lower_qns, 'triple2': upper_qns}) # Keep dict keys as 'triple1', 'triple2' for now

                except (ValueError, IndexError) as e:
                    print(f"Skipping invalid data format on line {line_num} in {calculated_file}: {line.strip()} - {str(e)}")
                    continue
                
    except NameError as e:
        print(f"Error: Required global variable might not be defined. {e}")
        messagebox.showerror("Setup Error", f"Required global variable might not be defined: {e}")
        return # Cannot proceed without CALC_SCALE
    except FileNotFoundError:
         print(f"Error: Calculated linelist file '{calculated_file}' not found.")
         messagebox.showerror("File Error", f"Calculated linelist file '{calculated_file}' not found.")
         return
    except Exception as e:
        print(f"Error reading calculated linelist file '{calculated_file}': {str(e)}")
        messagebox.showerror("File Error", f"Error reading calculated linelist file '{calculated_file}': {str(e)}")
        return

    if not entries:
        print("Warning: No valid entries loaded from calculated file.")
        messagebox.showwarning("Data Warning", "No valid entries loaded from calculated file.")
        # Don't return here, allow dialog to open anyway

    # --- Determine Root Window ---
    tk_root_required = True
    parent_window = None
    try:
        fig_manager = plt.get_current_fig_manager()
        if fig_manager is not None and hasattr(fig_manager, 'window') and isinstance(fig_manager.window, tk.Tk):
             if fig is not None and ax1 is not None and fig.canvas.manager == fig_manager:
                 print("Info: Using existing Matplotlib Tk window as parent.")
                 parent_window = fig_manager.window
                 tk_root_required = False
             else:
                 print("Warning: Active Matplotlib window found, but global 'fig'/'ax1' mismatch or None. Creating separate Tk context.")
                 fig, ax1 = None, None
        else:
             print("Info: No active Matplotlib Tk window found or suitable manager unavailable.")
    except Exception as e:
        print(f"Warning: Error checking figure manager ({e}). Creating separate Tk context.")

    if tk_root_required:
        print("Info: Creating standalone Tk root for dialog.")
        try:
            existing_root = tk._get_default_root('Error: Cannot determine default root Tk window')
            if existing_root:
                parent_window = Toplevel()
                print("Info: Using Toplevel as parent relative to existing Tk root.")
                tk_root_required = False
            else:
                 raise RuntimeError("No default root")
        except Exception:
             parent_window = tk.Tk()
             parent_window.withdraw()
             print("Info: Created new hidden Tk root.")

        if fig is None or ax1 is None:
              print("Warning: Plotting/Interaction will be disabled as 'fig'/'ax1' are None or no plot window is available.")
              fig, ax1 = None, None

    # --- Create the Main Dialog Window ---
    dialog = Toplevel(parent_window)
    dialog.title("Linelist Search, Evaluate & Loomis-Wood")
    position_dialog_near_cursor(dialog, width=800, height=600)

    # --- Input frame ---
    input_frame = Frame(dialog)
    input_frame.pack(pady=10, padx=10, fill='x')
    Label(input_frame, text="Enter upper state label").pack(side='left')
    input_entry = Entry(input_frame, width=30)
    input_entry.pack(side='left', padx=5)
    if 'destination_numbers' in globals() and destination_numbers is not None:
        try: input_entry.insert(0, ' '.join(map(str, destination_numbers)))
        except (TypeError, ValueError):
            print("Warning: Could not pre-fill input, 'destination_numbers' has unexpected type or format.")
            destination_numbers = None

    # --- Status Label (defined early for use in functions) ---
    results_frame = Frame(dialog)
    results_frame.pack(pady=5, padx=10, fill='x')
    stats_label = Label(results_frame, text="Enter an upper state triple and click Search, or use other functions.", justify=tk.LEFT, wraplength=780)
    stats_label.pack(fill='x')

    def Process_data(output_filename):
        # (Implementation as provided in the original question)
        # --- PHASE 1 AND PHASE 2 LOGIC REMAINS UNCHANGED ---
        global STATE_INDEX, ASSIGNMENT_BLOCK_SIZE
        ASSIGNMENT_BLOCK_SIZE = 2 * STATE_INDEX
        peaklist_file = peaklist_filename
        peaklist_lines_processed_count = 0
        peaklist_output_lines_written_count = 0
        peaklist_errors = []
        status_msg_part1 = ""
        try:
            with open(output_filename, 'w') as f_out:
                if not os.path.exists(peaklist_file):
                    error_msg = f"Warning: {peaklist_file} not found. Cannot process assignments."
                    print(error_msg)
                    peaklist_errors.append(error_msg)
                else:
                    print(f"Processing assignments from {peaklist_file} into {output_filename}...")
                    with open(peaklist_file, 'r') as f_peak:
                        for line_num, line in enumerate(f_peak, 1):
                            stripped_line = line.strip()
                            if not stripped_line: continue
                            parts = stripped_line.split()
                            num_cols = len(parts)
                            contour_type = parts[3]
                            if contour_type == 'Voigt':
                                assignment_start_index = 7
                            else: # Gaussian or Lorentzian
                                assignment_start_index = 6

                            # Check if there are enough columns for the base parameters + at least one assignment
                            if num_cols < assignment_start_index + ASSIGNMENT_BLOCK_SIZE:
                                continue

                            base_parts = parts[:assignment_start_index] # Base now includes all params
                            assignment_parts = parts[assignment_start_index:]
                            
                            num_assignment_cols = len(assignment_parts)
                            if num_assignment_cols % ASSIGNMENT_BLOCK_SIZE != 0:
                                error_msg = (f"Error: Invalid assignment format on line {line_num} in {peaklist_file}. "
                                            f"Assignment columns ({num_assignment_cols}) not multiple of {ASSIGNMENT_BLOCK_SIZE}. "
                                            f"Line: '{stripped_line}'")
                                print(error_msg)
                                peaklist_errors.append(error_msg)
                                continue
                            num_groups = num_assignment_cols // ASSIGNMENT_BLOCK_SIZE
                            peaklist_lines_processed_count += 1
                            for i in range(num_groups):
                                start_index = i * ASSIGNMENT_BLOCK_SIZE
                                end_index = start_index + ASSIGNMENT_BLOCK_SIZE
                                current_group = assignment_parts[start_index:end_index]
                                formatted_base = " ".join(base_parts)
                                output_line = f"{formatted_base} {' '.join(current_group)}\n"
                                f_out.write(output_line)
                                peaklist_output_lines_written_count += 1
            status_msg_part1 = (f"Processed {peaklist_lines_processed_count} lines from {peaklist_file}, "
                                f"wrote {peaklist_output_lines_written_count} assignment lines to {output_filename}.")
            if peaklist_errors:
                status_msg_part1 += f"\nEncountered {len(peaklist_errors)} errors during assignment processing (see console)."
            print(status_msg_part1)
        except Exception as e:
            error_msg = f"CRITICAL Error during Peaklist processing stage: {str(e)}"
            print(error_msg)
            try:
                if 'stats_label' in locals() or 'stats_label' in globals():
                    stats_label.config(text=error_msg)
            except Exception:
                pass
            return
        upper_energies_file = 'Upper.txt'
        processed_upper_states_count = 0
        upper_energy_errors = 0
        expected_upper_cols = 1 + STATE_INDEX
        if not os.path.exists(upper_energies_file):
            final_status_msg = f"{status_msg_part1}\nError: {upper_energies_file} not found. Cannot append upper energies."
            print(f"Error: {upper_energies_file} not found.")
            try:
                if 'stats_label' in locals() or 'stats_label' in globals():
                    stats_label.config(text=final_status_msg)
            except Exception: pass
            return
        data = []
        try:
            with open(upper_energies_file, 'r') as f:
                line_count = 0
                for line in f:
                    line_count += 1
                    parts = line.strip().split()
                    if len(parts) < expected_upper_cols:
                        if len(parts) > 0:
                            print(f"Skipping malformed line in {upper_energies_file} line {line_count} (expected >= {expected_upper_cols} parts, got {len(parts)}): {line.strip()}")
                        continue
                    try:
                        energy = float(parts[0])
                        qns = tuple(map(int, parts[1:1+STATE_INDEX]))
                        if len(qns) != STATE_INDEX:
                            raise ValueError(f"Incorrect number of QNs found ({len(qns)}), expected {STATE_INDEX}")
                        data.append( (energy, qns) )
                    except (ValueError, IndexError) as e:
                        print(f"Skipping non-numeric/format error in {upper_energies_file} line {line_count}: {line.strip()} - {e}")
                        upper_energy_errors += 1
                        continue
        except Exception as e:
            final_status_msg = f"{status_msg_part1}\nError reading {upper_energies_file}: {str(e)}"
            print(f"Error reading {upper_energies_file}: {str(e)}")
            try:
                if 'stats_label' in locals() or 'stats_label' in globals():
                    stats_label.config(text=final_status_msg)
            except Exception: pass
            return
        if upper_energy_errors > 0:
            print(f"Warning: Encountered {upper_energy_errors} formatting errors while reading QNs from {upper_energies_file}.")
        if not data:
            final_status_msg = f"{status_msg_part1}\n{upper_energies_file} is empty or contains no valid data. No upper energies appended."
            print(f"{upper_energies_file} is empty or contains no valid data.")
            try:
                if 'stats_label' in locals() or 'stats_label' in globals():
                    stats_label.config(text=final_status_msg)
            except Exception: pass
            return
        unique_qns = sorted(list(set(qn_tuple for energy, qn_tuple in data)))
        processed = []
        for qn_tuple in unique_qns:
            energies = [e for e, t in data if t == qn_tuple]
            if energies:
                avg = np.mean(energies)
                std_dev = np.std(energies) if len(energies) > 1 else 0.0
                processed.append( (qn_tuple, avg, std_dev) )
        processed.sort(key=lambda item: item[1])
        processed_upper_states_count = len(processed)
        try:
            with open(output_filename, 'a') as f_out:
                write_count = 0
                for qn_tuple, avg, std in processed:
                    if len(qn_tuple) == STATE_INDEX:
                        qn_str = " ".join(map(str, qn_tuple))
                        f_out.write(f"{avg:.8f} {qn_str} {std:.8f}\n")
                        write_count += 1
                    else:
                        print(f"Error: Skipping write to {output_filename} - Incorrect QN tuple length {len(qn_tuple)} for avg energy {avg}. Expected {STATE_INDEX}.")
                final_status_msg = (f"{status_msg_part1}\nAppended {write_count} "
                                    f"processed upper states (sorted by avg energy) from {upper_energies_file} "
                                    f"to {output_filename}.")
            print(f"Successfully appended {processed_upper_states_count} processed upper states to {output_filename}")
            try:
                if 'stats_label' in locals() or 'stats_label' in globals():
                    stats_label.config(text=final_status_msg, justify=tk.LEFT)
            except Exception as e:
                print(f"Warning: Failed to update final status label: {e}")
        except Exception as e:
            error_msg = f"Error appending upper energy data to {output_filename}: {str(e)}"
            print(error_msg)
            final_status_msg = f"{status_msg_part1}\n{error_msg}"
            try:
                if 'stats_label' in locals() or 'stats_label' in globals():
                    stats_label.config(text=final_status_msg, justify=tk.LEFT)
            except Exception: pass

        # ==============================================================================
        # --- PHASE 3: CREATE PUBLICATION-STYLE ASSIGNMENT LIST ---
        # This new section reads Peaklist, groups assignments by peak,
        # sorts them by frequency, and formats them as shown in the user's image.
        # ==============================================================================
        print("\n--- Starting Phase 3: Generating Publication-Style Assignment List ---")
        try:
            # Step 1: Read Peaklist and group assignments by peak (freq, integral).
            # A dictionary is perfect for this: { (freq, integral): [list_of_assignments] }
            peak_assignments_map = {}
            with open(peaklist_file, 'r') as f_peak:
                for line in f_peak:
                    parts = line.strip().split()
                    if len(parts) < 5: continue

                    try:
                        peak_freq = float(parts[1])
                        contour_type = parts[3]

                        # Determine the correct index for the integral and assignments
                        if contour_type == 'Voigt':
                            integral_index = 6
                            assignment_start_index = 7
                        else:  # Gaussian or Lorentzian
                            integral_index = 5
                            assignment_start_index = 6
                        
                        if len(parts) <= integral_index: continue
                        peak_integral = float(parts[integral_index])
                        
                        # Create a unique key for this peak
                        peak_key = (peak_freq, peak_integral)

                        # Unpack all assignments for this line
                        assignment_parts = parts[assignment_start_index:]
                        if len(assignment_parts) % ASSIGNMENT_BLOCK_SIZE == 0:
                            if peak_key not in peak_assignments_map:
                                peak_assignments_map[peak_key] = []
                            
                            for i in range(0, len(assignment_parts), ASSIGNMENT_BLOCK_SIZE):
                                assignment_chunk = assignment_parts[i : i + ASSIGNMENT_BLOCK_SIZE]
                                peak_assignments_map[peak_key].append(assignment_chunk)

                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue
            
            # Step 2: Sort the peaks by frequency.
            sorted_peaks = sorted(peak_assignments_map.keys())

            # Step 3: Write the formatted data to the output file.
            if sorted_peaks:
                with open(output_filename, 'a') as f_out:
                    f_out.write("\n\n# --- Publication-Style Assignments (Sorted by Frequency) ---\n")
                    f_out.write("# Frequency      Integral      Lower State -> Upper State\n")
                    
                    for peak_key in sorted_peaks:
                        peak_freq, peak_integral = peak_key
                        all_assignments_for_peak = peak_assignments_map[peak_key]

                        if not all_assignments_for_peak: continue

                        # Handle the first assignment on the same line
                        first_assignment = all_assignments_for_peak[0]
                        lower_qns_str = " ".join(first_assignment[0:STATE_INDEX])
                        upper_qns_str = " ".join(first_assignment[STATE_INDEX:ASSIGNMENT_BLOCK_SIZE])
                        
                        # Write the main line for the peak
                        f_out.write(f"{peak_freq:<14.6f} {peak_integral:<13.6e} {lower_qns_str} {upper_qns_str}\n")

                        # Handle subsequent assignments on new, indented lines
                        for other_assignment in all_assignments_for_peak[1:]:
                            lower_qns_str = " ".join(other_assignment[0:STATE_INDEX])
                            upper_qns_str = " ".join(other_assignment[STATE_INDEX:ASSIGNMENT_BLOCK_SIZE])
                            
                            # Use f-string formatting to create the indentation
                            indentation = " " * 28
                            f_out.write(f"{indentation}{lower_qns_str} {upper_qns_str}\n")

                print(f"Phase 3: Successfully appended formatted assignment list for {len(sorted_peaks)} unique peaks.")
                final_status_msg += f"\nAppended publication-style list for {len(sorted_peaks)} peaks."
            else:
                print("Phase 3: No valid assignments found in Peaklist to format.")
                final_status_msg += "\nNo assignments found to create publication-style list."

        except FileNotFoundError:
            print(f"Phase 3 Error: {peaklist_file} not found. Cannot generate formatted list.")
            final_status_msg += f"\nError: {peaklist_file} not found for Phase 3."
        except Exception as e:
            print(f"CRITICAL Error during Phase 3 generation: {e}")
            final_status_msg += f"\nError during publication list generation: {e}"

        # ==============================================================================
        # --- NEW PHASE 4: Generate Sorted List by Contour and Frequency ---
        # This new section reads the database, sorts it by contour then by peak center,
        # and appends a list without quantum number assignments.
        # ==============================================================================
        print("\n--- Starting Phase 4: Generating Sorted List by Contour and Frequency ---")
        try:
            peaks_for_sorting = []
            with open(peaklist_file, 'r') as f_peak:
                for line in f_peak:
                    parts = line.strip().split()
                    if len(parts) < 4: continue # Must have at least symbol, center, amp, contour

                    try:
                        peak_center = float(parts[1])
                        contour_type = parts[3]

                        # Determine where the data ends and assignments begin
                        assignment_start_index = 0
                        if contour_type == 'Voigt':
                            # Symbol, Center, Amp, Contour, FWHM_G, FWHM_L, Integral
                            assignment_start_index = 7
                        else:  # Gaussian or Lorentzian
                            # Symbol, Center, Amp, Contour, FWHM, Integral
                            assignment_start_index = 6

                        # Ensure the line has enough base parts before trying to slice
                        if len(parts) >= assignment_start_index:
                            base_peak_data = parts[:assignment_start_index]
                            line_without_assignments = " ".join(base_peak_data)
                            peaks_for_sorting.append((contour_type, peak_center, line_without_assignments))

                    except (ValueError, IndexError):
                        # Skip malformed lines gracefully
                        continue
            
            # Sort the collected data: primary key is contour_type (string), secondary is peak_center (float)
            sorted_peaks_data = sorted(peaks_for_sorting, key=lambda item: (item[0], item[1]))

            # Write the formatted data to the output file
            if sorted_peaks_data:
                with open(output_filename, 'a') as f_out:
                    f_out.write("\n\n# --- Peaks Sorted by Contour and Frequency (Assignments Removed) ---\n")
                    f_out.write("# Contour      Peak Data (Symbol, Center, Amplitude, FWHM(s), Integral)\n")
                    
                    current_contour = None
                    for contour, center, line_data in sorted_peaks_data:
                        # Add a separator when the contour type changes for readability
                        if contour != current_contour:
                            if current_contour is not None:
                                f_out.write("\n") # Add a blank line between groups
                            f_out.write(f"# --- {contour} ---\n")
                            current_contour = contour
                        
                        f_out.write(line_data + "\n")

                print(f"Phase 4: Successfully appended sorted list for {len(sorted_peaks_data)} peaks.")
                final_status_msg += f"\nAppended contour-sorted list for {len(sorted_peaks_data)} peaks."
            else:
                print("Phase 4: No valid peaks found in Peaklist to sort and append.")
                final_status_msg += "\nNo peaks found to create contour-sorted list."

        except FileNotFoundError:
            print(f"Phase 4 Error: {peaklist_file} not found. Cannot generate sorted list.")
            final_status_msg += f"\nError: {peaklist_file} not found for Phase 4."
        except Exception as e:
            print(f"CRITICAL Error during Phase 4 generation: {e}")
            final_status_msg += f"\nError during contour-sorted list generation: {e}"


        # Update the GUI status label with the final summary
        try:
            if 'stats_label' in locals() or 'stats_label' in globals():
                stats_label.config(text=final_status_msg, justify=tk.LEFT)
        except Exception as e:
            print(f"Warning: Failed to update final status label after Phase 4: {e}")

    def perform_search():
        nonlocal filtered_entries
        global destination_numbers
        input_str = input_entry.get().strip()
        try:
            target_qns_str = input_str.split()
            if len(target_qns_str) != STATE_INDEX:
                raise ValueError(f"Exactly {STATE_INDEX} integers required for upper state")
            target_qns_tuple = tuple(map(int, target_qns_str))
        except Exception as e:
            stats_label.config(text=f"Invalid input: {str(e)}")
            filtered_entries = []
            listbox.delete(0, 'end')
            return
        try:
            with open("LabelsTemp.txt", 'w') as f_clear: pass
        except Exception as e:
            print(f"Warning: Could not clear LabelsTemp.txt on search: {e}")
        destination_numbers = target_qns_tuple
        filtered_entries = [entry for entry in entries if entry['triple2'] == target_qns_tuple]
        filtered_entries.sort(key=lambda entry: entry['intensity'], reverse=True)
        listbox.delete(0, 'end')
        if not filtered_entries:
            target_qns_display = " ".join(map(str, target_qns_tuple))
            stats_label.config(text=f"No entries found matching upper state: {target_qns_display}")
        else:
            target_qns_display = " ".join(map(str, target_qns_tuple))
            stats_label.config(text=f"Found {len(filtered_entries)} entries matching upper state {target_qns_display} (sorted by Intensity). Click to plot.")
            for entry in filtered_entries:
                lower_qns_display = " ".join(map(str, entry['triple1']))
                listbox.insert('end', f"Freq: {entry['freq']:.8f} | Int: {entry['intensity']:.6e} | Lower: {lower_qns_display}")

    def on_select(event):
        nonlocal current_line
        if ax1 is None or fig is None:
            print("Plotting disabled (no valid fig/ax1). Cannot plot selection.")
            stats_label.config(text="Plotting disabled (no active plot).")
            return
        selection = listbox.curselection()
        if not selection: return
        index = selection[0]
        if index >= len(filtered_entries): return
        try:
             entry = filtered_entries[index]
             freq = entry['freq']
             intensity = entry['intensity']
             lower_qns = entry['triple1']
             upper_qns = entry['triple2']
        except KeyError as e:
             print(f"Error: Missing key {e} in selected entry: {entry}")
             stats_label.config(text="Error: Corrupted data for selection.")
             return
        except Exception as e:
            print(f"Error accessing filtered entry data at index {index}: {e}")
            stats_label.config(text="Error: Could not read selected data.")
            return
        _write_selection_to_labels_temp(lower_qns, upper_qns)
        line_just_removed = False
        if current_line is not None:
            try:
                if current_line in ax1.lines:
                   current_line.remove()
                   line_just_removed = True
                current_line = None
            except Exception as e:
                 print(f"Warning: Error removing previous line: {e}")
                 current_line = None
        try:
            y_lim = ax1.get_ylim()
            plot_intensity = abs(intensity) if intensity != 0 else 1e-9
            plot_y_end = min(plot_intensity, y_lim[1])
            plot_y_start = 0.0
            if y_lim[1] <= y_lim[0]:
                 plot_y_start = 0; plot_y_end = plot_intensity
            elif plot_y_end <= y_lim[0]:
                 plot_y_start = y_lim[0]
                 plot_y_end = y_lim[0] + 0.05 * (y_lim[1] - y_lim[0])
            elif plot_y_start >= plot_y_end:
                 plot_y_start = y_lim[0]
                 plot_y_end = y_lim[0] + 0.05 * (y_lim[1] - y_lim[0])
            line_label = 'Selected Calc' if line_just_removed or current_line is None else ""
            current_line = ax1.vlines(freq, plot_y_start, plot_y_end,
                                      colors='red', linestyles='--', linewidth=1.5,
                                      label=line_label, zorder=10)
            lower_qns_display = " ".join(map(str, lower_qns))
            upper_qns_display = " ".join(map(str, upper_qns))
            stats_label.config(text=f"Selected: F={freq:.6f}, I={intensity:.3e} | Transition: {lower_qns_display} → {upper_qns_display}")
            if fig and ax1:
                current_xlim = ax1.get_xlim()
                if current_xlim[1] > current_xlim[0]: x_range = current_xlim[1] - current_xlim[0]
                else: x_range = freq * 0.1 if freq > 0 else 10.0
                new_xlim = (freq - x_range / 2, freq + x_range / 2)
                min_x_range = 1e-5
                if not all(np.isfinite(x) for x in new_xlim) or abs(new_xlim[1] - new_xlim[0]) < min_x_range:
                     new_xlim = (freq - min_x_range*5, freq + min_x_range*5)
                ax1.set_xlim(new_xlim)
                ax2.set_xlim(new_xlim)
                fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error during plotting or redraw in on_select: {e}")
            messagebox.showerror("Plot Error", f"Error updating plot: {e}", parent=dialog)
            current_line = None

    def perform_evaluate():
        global STATE_INDEX, ASSIGNMENT_BLOCK_SIZE
        global destination_numbers
        import numpy as np
        import os
        ASSIGNMENT_BLOCK_SIZE = 2 * STATE_INDEX
        expected_peaklist_min_cols = 6 + ASSIGNMENT_BLOCK_SIZE
        expected_aux_cols = ASSIGNMENT_BLOCK_SIZE + 2
        input_str = input_entry.get().strip()
        try:
            target_qns_str = input_str.split()
            if len(target_qns_str) != STATE_INDEX:
                raise ValueError(f"Exactly {STATE_INDEX} integers required for upper state")
            current_destination_numbers = tuple(map(int, target_qns_str))
            destination_numbers = current_destination_numbers
        except Exception as e:
            stats_label.config(text=f"Invalid input for Evaluate: {str(e)}")
            return
        peaklist_file = peaklist_filename
        auxil_peaklist_file = 'auxil_peaklist.txt'
        peaklist_lines_processed = 0
        peaklist_errors = 0
        print(f"Evaluate: Processing {peaklist_file} -> {auxil_peaklist_file} (using STATE_INDEX={STATE_INDEX})")
        try:
            with open(peaklist_file, 'r') as peakfile, open(auxil_peaklist_file, 'w') as auxfile:
                line_count = 0
                for line in peakfile:
                    line_count += 1
                    parts = line.strip().split()
                    if len(parts) < expected_peaklist_min_cols:
                        continue
                    try:
                        peak_freq = float(parts[1])
                        peak_int = float(parts[2])
                        contour_type = parts[3]
                        if contour_type == 'Voigt':
                            assignment_start_index = 7
                        else: # Gaussian or Lorentzian
                            assignment_start_index = 6
                        numbers_str = parts[assignment_start_index:]
                        if len(numbers_str) % ASSIGNMENT_BLOCK_SIZE != 0:
                             print(f"Evaluate (Write): Skipping assignment on line {line_count} in {peaklist_file}: Incorrect number of values ({len(numbers_str)}).")
                             peaklist_errors += 1
                             continue
                        
                        numbers = list(map(int, numbers_str))
                        valid_assignment_found_in_line = False
                        for i in range(0, len(numbers), ASSIGNMENT_BLOCK_SIZE):
                            group = numbers[i:i+ASSIGNMENT_BLOCK_SIZE]
                            qn_group_str = " ".join(map(str, group))
                            auxfile.write(f"{qn_group_str} {peak_freq:.8f} {peak_int:.6e}\n")
                            valid_assignment_found_in_line = True
                        if valid_assignment_found_in_line:
                            peaklist_lines_processed += 1
                    except (ValueError, IndexError) as e:
                        print(f"Evaluate (Write): Skipping non-numeric data in {peaklist_file} line {line_count}: {line.strip()} - {e}")
                        peaklist_errors += 1
                        continue
            print(f"Evaluate: Wrote {peaklist_lines_processed} lines with assignments to {auxil_peaklist_file}.")
            if peaklist_errors > 0: print(f"Evaluate: Encountered {peaklist_errors} errors during writing.")
        except FileNotFoundError:
            stats_label.config(text=f"Error: {peaklist_file} not found.")
            if os.path.exists(auxil_peaklist_file):
                try: os.remove(auxil_peaklist_file)
                except OSError as e_rem: print(f"Warning: Could not remove {auxil_peaklist_file}: {e_rem}")
            return
        except Exception as e:
            stats_label.config(text=f"Error processing {peaklist_file}: {str(e)}")
            if os.path.exists(auxil_peaklist_file):
                try: os.remove(auxil_peaklist_file)
                except OSError as e_rem: print(f"Warning: Could not remove {auxil_peaklist_file}: {e_rem}")
            return
        print(f"Evaluate: Processed {peaklist_lines_processed} lines from {peaklist_file} into {auxil_peaklist_file}.")
        aux_entries = []
        print(f"Evaluate: Reading {auxil_peaklist_file} (expecting {expected_aux_cols} columns)")
        try:
            with open(auxil_peaklist_file, 'r') as auxfile:
                line_count = 0
                for line in auxfile:
                     line_count += 1
                     parts = line.strip().split()
                     if len(parts) != expected_aux_cols:
                         print(f"Evaluate (Read): Skipping malformed line in {auxil_peaklist_file} line {line_count} (expected {expected_aux_cols} parts, got {len(parts)}): {line.strip()}")
                         continue
                     try:
                         lower_qns = tuple(map(int, parts[0:STATE_INDEX]))
                         upper_qns = tuple(map(int, parts[STATE_INDEX:ASSIGNMENT_BLOCK_SIZE]))
                         freq = float(parts[ASSIGNMENT_BLOCK_SIZE])
                         intensity = float(parts[ASSIGNMENT_BLOCK_SIZE + 1])
                         aux_entries.append( (lower_qns, upper_qns, freq, intensity) )
                     except (ValueError, IndexError) as e:
                         print(f"Evaluate (Read): Skipping non-numeric data in {auxil_peaklist_file} line {line_count}: {line.strip()} - {e}")
                         continue
            print(f"Evaluate: Read {len(aux_entries)} transitions from {auxil_peaklist_file}.")
        except FileNotFoundError:
            stats_label.config(text=f"Error: {auxil_peaklist_file} not found (should have been created).")
            return
        except Exception as e:
            stats_label.config(text=f"Error reading {auxil_peaklist_file}: {str(e)}")
            if os.path.exists(auxil_peaklist_file):
                try: os.remove(auxil_peaklist_file)
                except OSError as e_rem: print(f"Warning: Could not remove {auxil_peaklist_file}: {e_rem}")
            return
        lower_energies_dict = {}
        lower_energy_duplicates = 0
        lower_energy_qn_errors = 0
        expected_lower_cols = 1 + STATE_INDEX
        try:
            with open(lower_energies_file, 'r') as lowerfile:
                 line_count = 0
                 for line in lowerfile:
                    line_count += 1
                    parts = line.strip().split()
                    if len(parts) < expected_lower_cols:
                        if len(parts) > 0:
                            print(f"Skipping malformed line in {lower_energies_file} line {line_count} (expected >= {expected_lower_cols} parts, got {len(parts)}): {line.strip()}")
                        continue
                    try:
                        energy = float(parts[0])
                        qns = tuple(map(int, parts[1:1+STATE_INDEX]))
                        if len(qns) != STATE_INDEX:
                             raise ValueError(f"Incorrect number of QNs found ({len(qns)}), expected {STATE_INDEX}")
                        if qns in lower_energies_dict:
                             lower_energy_duplicates += 1
                        lower_energies_dict[qns] = energy
                    except (ValueError, IndexError) as e:
                         print(f"Skipping non-numeric/format error in {lower_energies_file} line {line_count}: {line.strip()} - {e}")
                         lower_energy_qn_errors += 1
                         continue
        except FileNotFoundError:
            stats_label.config(text=f"Error: {lower_energies_file} not found.")
            if os.path.exists(auxil_peaklist_file):
                try: os.remove(auxil_peaklist_file)
                except OSError as e_rem: print(f"Warning: Could not remove {auxil_peaklist_file}: {e_rem}")
            return
        except Exception as e:
            stats_label.config(text=f"Error reading {lower_energies_file}: {str(e)}")
            if os.path.exists(auxil_peaklist_file):
                try: os.remove(auxil_peaklist_file)
                except OSError as e_rem: print(f"Warning: Could not remove {auxil_peaklist_file}: {e_rem}")
            return
        if lower_energy_qn_errors > 0:
            print(f"Warning: Encountered {lower_energy_qn_errors} formatting errors while reading QNs from {lower_energies_file}.")
        if lower_energy_duplicates > 0:
             print(f"Warning: Found {lower_energy_duplicates} duplicate states in {lower_energies_file}; used the last occurrence for calculations.")
        print(f"Evaluate: Read {len(lower_energies_dict)} unique lower state energies from {lower_energies_file}.")
        matching_aux_for_stats = [entry for entry in aux_entries if entry[1] == current_destination_numbers]
        upper_empir_for_stats = []
        count_lower_found_for_stats = 0
        count_lower_missing_for_stats = 0
        if matching_aux_for_stats:
            for lower_qns, _, freq, _ in matching_aux_for_stats:
                if lower_qns in lower_energies_dict:
                    calculated_upper_energy_for_stats = lower_energies_dict[lower_qns] + freq
                    upper_empir_for_stats.append(calculated_upper_energy_for_stats)
                    count_lower_found_for_stats += 1
                else:
                    count_lower_missing_for_stats += 1
        target_qns_display = " ".join(map(str, current_destination_numbers))
        stats_text = f"Stats for Upper State {target_qns_display}:\n"
        if not upper_empir_for_stats:
             stats_text += f"No energies calculated. Found {len(matching_aux_for_stats)} transitions ending in this state, "
             if len(matching_aux_for_stats) > 0:
                  stats_text += f"but their lower states were not found in {lower_energies_file} ({count_lower_missing_for_stats} missing)."
             else:
                  stats_text += f"and no transitions ending in this state were found in {peaklist_file}."
        else:
            count = len(upper_empir_for_stats)
            avg = np.mean(upper_empir_for_stats)
            std_dev = np.std(upper_empir_for_stats) if count > 1 else 0.0
            min_val = np.min(upper_empir_for_stats)
            max_val = np.max(upper_empir_for_stats)
            range_val = max_val - min_val
            stats_text += (f"Determined from {count} transitions with known lower states:\n"
                           f"  Avg: {avg:.8f} | StdDev: {std_dev:.8f}\n"
                           f"  Min: {min_val:.8f} | Max: {max_val:.8f} | Range: {range_val:.8f}\n")
            if count_lower_missing_for_stats > 0:
                stats_text += f"({count_lower_missing_for_stats} additional transitions ignored due to unknown lower state energy)."
        all_individual_upper_entries = []
        processed_transitions_count = 0
        lower_state_needed_count = 0
        for aux_lower_qns, aux_upper_qns, aux_freq, _ in aux_entries:
            if aux_lower_qns in lower_energies_dict:
                calculated_upper_energy = lower_energies_dict[aux_lower_qns] + aux_freq
                all_individual_upper_entries.append( (calculated_upper_energy, aux_upper_qns) )
                processed_transitions_count += 1
            else:
                 lower_state_needed_count += 1
        print(f"Evaluate: Calculated {len(all_individual_upper_entries)} individual upper energy determinations "
              f"from {processed_transitions_count} transitions with known lower states. "
              f"{lower_state_needed_count} transitions skipped (missing lower state).")
        all_individual_upper_entries.sort(key=lambda x: x[0])
        try:
            with open(upper_energies_file, 'w') as upperfile:
                if not all_individual_upper_entries:
                    print(f"Warning: No upper energies could be calculated to write to {upper_energies_file}.")
                else:
                    write_count = 0
                    for energy, qn_tuple in all_individual_upper_entries:
                        if len(qn_tuple) == STATE_INDEX:
                             qn_str = " ".join(map(str, qn_tuple))
                             upperfile.write(f"{energy:.8f} {qn_str}\n")
                             write_count += 1
                        else:
                             print(f"Error: Skipping write to {upper_energies_file} - Incorrect QN tuple length {len(qn_tuple)} for energy {energy}. Expected {STATE_INDEX}.")
            stats_text += f"\n\n{upper_energies_file} regenerated with {write_count} individual upper state energy determinations, sorted by energy."
        except Exception as e:
            stats_text += f"\n\nCRITICAL ERROR writing updated {upper_energies_file}: {str(e)}"
            stats_label.config(text=stats_text, justify=tk.LEFT)
            if os.path.exists(auxil_peaklist_file):
                try: os.remove(auxil_peaklist_file)
                except Exception as e_rem: print(f"Warning: Could not remove {auxil_peaklist_file} after write error: {e_rem}")
            return
        if os.path.exists(auxil_peaklist_file):
            try:
                os.remove(auxil_peaklist_file)
                print(f"Evaluate: Removed temporary file: {auxil_peaklist_file}")
            except Exception as e:
                stats_text += f"\nWarning: Could not remove temporary file {auxil_peaklist_file}: {str(e)}"
        stats_label.config(text=stats_text, justify=tk.LEFT)

    def _process_peaklist_for_lw(peaklist_filename, output_auxil_filename):
        global STATE_INDEX, ASSIGNMENT_BLOCK_SIZE
        peaklist_lines_processed = 0
        peaklist_errors = 0
        expected_min_peaklist_cols = 6 + ASSIGNMENT_BLOCK_SIZE
        print(f"LW: Processing {peaklist_filename} -> {output_auxil_filename}")
        try:
            with open(peaklist_filename, 'r') as peakfile, open(output_auxil_filename, 'w') as auxfile:
                line_count = 0
                for line in peakfile:
                    line_count += 1
                    parts = line.strip().split()
                    if len(parts) < expected_min_peaklist_cols: continue
                    try:

                        peak_freq = float(parts[1])
                        peak_int = float(parts[2])

                        contour_type = parts[3]
                        if contour_type == 'Voigt':
                            assignment_start_index = 7
                        else: # Gaussian or Lorentzian
                            assignment_start_index = 6

                        numbers_str = parts[assignment_start_index:]

                        if len(numbers_str) % ASSIGNMENT_BLOCK_SIZE != 0:
                            print(f"LW: Skipping assignment on line {line_count} in {peaklist_filename}: Incorrect number of assignment values ({len(numbers_str)}).")
                            peaklist_errors += 1
                            continue


                        numbers = list(map(int, numbers_str))
                        valid_assignment_found_in_line = False
                        for i in range(0, len(numbers), ASSIGNMENT_BLOCK_SIZE):
                            group = numbers[i:i+ASSIGNMENT_BLOCK_SIZE]
                            qn_group_str = " ".join(map(str, group))
                            auxfile.write(f"{qn_group_str} {peak_freq:.8f} {peak_int:.6e}\n")
                            valid_assignment_found_in_line = True
                        if valid_assignment_found_in_line:
                            peaklist_lines_processed += 1
                    except (ValueError, IndexError) as e:
                        print(f"LW: Skipping non-numeric data in {peaklist_filename} line {line_count}: {line.strip()} - {e}")
                        peaklist_errors += 1
                        continue
            print(f"LW: Successfully processed {peaklist_lines_processed} lines with assignments from {peaklist_filename} into {output_auxil_filename}.")
            if peaklist_errors > 0: print(f"LW: Encountered {peaklist_errors} errors during processing.")
            return True
        except FileNotFoundError:
            messagebox.showerror("LW Error", f"Experimental peaklist file '{peaklist_filename}' not found.", parent=lw_window if lw_window else dialog)
            if os.path.exists(output_auxil_filename):
                try: os.remove(output_auxil_filename)
                except OSError as e_rem: print(f"Warning: Could not remove {output_auxil_filename}: {e_rem}")
            return False
        except Exception as e:
            messagebox.showerror("LW Error", f"Error processing {peaklist_filename} for LW plot: {str(e)}", parent=lw_window if lw_window else dialog)
            if os.path.exists(output_auxil_filename):
                try: os.remove(output_auxil_filename)
                except OSError as e_rem: print(f"Warning: Could not remove {output_auxil_filename}: {e_rem}")
            return False

    def _read_auxil_for_lw(auxil_filename='auxil_peaklist_lw.txt'):
        global STATE_INDEX, ASSIGNMENT_BLOCK_SIZE
        exp_entries = []
        expected_aux_cols = ASSIGNMENT_BLOCK_SIZE + 2
        try:
            with open(auxil_filename, 'r') as auxfile:
                line_count = 0
                for line in auxfile:
                     line_count += 1
                     parts = line.strip().split()
                     if len(parts) != expected_aux_cols:
                         print(f"LW Read: Skipping malformed line in {auxil_filename} line {line_count} (expected {expected_aux_cols} parts): {line.strip()}")
                         continue
                     try:
                         lower_qns = tuple(map(int, parts[0:STATE_INDEX]))
                         upper_qns = tuple(map(int, parts[STATE_INDEX:ASSIGNMENT_BLOCK_SIZE]))
                         freq = float(parts[ASSIGNMENT_BLOCK_SIZE])
                         intensity = float(parts[ASSIGNMENT_BLOCK_SIZE + 1])
                         exp_entries.append({'freq': freq, 'intensity': intensity, 'triple1': lower_qns, 'triple2': upper_qns})
                     except (ValueError, IndexError) as e:
                         print(f"LW Read: Skipping non-numeric data in {auxil_filename} line {line_count}: {line.strip()} - {e}")
                         continue
            print(f"LW Read: Read {len(exp_entries)} experimental transitions from {auxil_filename}.")
        except FileNotFoundError:
            print(f"LW Info: Auxiliary experimental file '{auxil_filename}' not found (expected if checkbox unchecked or first run).")
            return []
        except Exception as e:
            messagebox.showerror("LW Error", f"Error reading auxiliary file {auxil_filename}: {str(e)}", parent=lw_window if lw_window else dialog)
            return []
        return exp_entries

    def _on_lw_window_close():
        nonlocal lw_window, lw_fig, lw_ax, lw_folding_entry, lw_intensity_entry, lw_ka_entry, lw_canvas
        nonlocal lw_plot_exp_var, lw_qn_index_entry
        print("Closing Loomis-Wood window.")
        if lw_window:
            lw_window.destroy()
        lw_window = None
        lw_fig = None
        lw_ax = None
        lw_folding_entry = None
        lw_intensity_entry = None
        lw_ka_entry = None
        lw_qn_index_entry = None
        lw_canvas = None
        lw_plot_exp_var = None
        loomis_wood_data_store.clear()
        loomis_wood_data_store_exp.clear()
        aux_lw_file = 'auxil_peaklist_lw.txt'
        if os.path.exists(aux_lw_file):
            try:
                os.remove(aux_lw_file)
                print(f"Removed temporary file: {aux_lw_file}")
            except OSError as e:
                print(f"Warning: Could not remove temporary file {aux_lw_file}: {e}")

    def _update_lw_plot_from_gui():
        nonlocal lw_folding_entry, lw_intensity_entry, lw_ka_entry, lw_window, lw_ax, lw_plot_exp_var
        nonlocal lw_qn_index_entry
        if not lw_window or not lw_ax or not lw_plot_exp_var or not lw_qn_index_entry:
            print("LW Error: Cannot update, window/axes/checkbox var/QN index entry not available.")
            return
        try:
            folding_const = float(lw_folding_entry.get())
            min_intensity = float(lw_intensity_entry.get())
            upper_qn_value_str = lw_ka_entry.get().strip()
            upper_qn_value_filter = None
            if upper_qn_value_str:
                upper_qn_value_filter = int(upper_qn_value_str)
            qn_filter_index_str = lw_qn_index_entry.get().strip()
            qn_filter_index_0_based = 1
            if qn_filter_index_str:
                try:
                    gui_index = int(qn_filter_index_str)
                    if gui_index <= 0:
                        raise ValueError("QN# must be a positive integer.")
                    qn_filter_index_0_based = gui_index - 1
                except ValueError as e_idx:
                    messagebox.showerror("Invalid Input", f"Error in QN#: {e_idx}", parent=lw_window)
                    return
            if folding_const <= 0: raise ValueError("Folding constant must be positive.")
            if min_intensity < 0: raise ValueError("Minimum intensity must be non-negative.")
            plot_experimental_flag = lw_plot_exp_var.get()
            plot_loomis_wood(lw_ax, folding_const, min_intensity, 
                             upper_qn_value_filter, qn_filter_index_0_based,
                             plot_experimental_flag)
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Error in parameters: {e}", parent=lw_window)
        except Exception as e:
             print(f"Unexpected error during LW plot update: {e}")
             messagebox.showerror("Error", f"An unexpected error occurred during update: {e}", parent=lw_window)

    def create_or_update_loomis_wood_window():
        nonlocal entries, lw_window, lw_fig, lw_ax, lw_folding_entry, lw_intensity_entry, lw_ka_entry, dialog, lw_canvas
        nonlocal lw_plot_exp_var, lw_qn_index_entry
        if not entries:
            messagebox.showwarning("No Data", "Cannot create Loomis-Wood diagram, no calculated linelist data loaded.", parent=dialog)
            return
        if lw_window and lw_window.winfo_exists():
            print("Loomis-Wood window already open. Lifting.")
            lw_window.lift()
            lw_window.attributes('-topmost', True)
            lw_window.after_idle(lw_window.attributes, '-topmost', False)
            return
        
        lw_window = Toplevel(dialog)
        lw_window.title("Loomis-Wood Diagram & Parameters")
        position_dialog_near_cursor(lw_window, width=850, height=650)

        lw_window.protocol("WM_DELETE_WINDOW", _on_lw_window_close)
        lw_plot_exp_var = BooleanVar(master=lw_window, value=False)
        control_frame = Frame(lw_window, padx=5, pady=5)
        control_frame.pack(side='top', fill='x')
        Label(control_frame, text="Folding Const:").grid(row=0, column=0, sticky='w', padx=2, pady=2)
        lw_folding_entry = Entry(control_frame, width=10)
        lw_folding_entry.grid(row=0, column=1, sticky='ew', padx=(0,5), pady=2)
        lw_folding_entry.insert(0, "0.1")
        Label(control_frame, text="Min Intensity:").grid(row=0, column=2, sticky='w', padx=2, pady=2)
        lw_intensity_entry = Entry(control_frame, width=10)
        lw_intensity_entry.grid(row=0, column=3, sticky='ew', padx=(0,5), pady=2)
        lw_intensity_entry.insert(0, "1e-9")
        Label(control_frame, text="Upper QN Value:").grid(row=0, column=4, sticky='w', padx=2, pady=2)
        lw_ka_entry = Entry(control_frame, width=5)
        lw_ka_entry.grid(row=0, column=5, sticky='ew', padx=(0,5), pady=2)
        lw_ka_entry.insert(0, "")
        Label(control_frame, text="Filter QN#:").grid(row=0, column=6, sticky='w', padx=2, pady=2)
        lw_qn_index_entry = Entry(control_frame, width=3)
        lw_qn_index_entry.grid(row=0, column=7, sticky='ew', padx=(0,10), pady=2)
        lw_qn_index_entry.insert(0, "2")
        exp_check = Checkbutton(control_frame, text="Plot Exp", variable=lw_plot_exp_var,
                                command=_update_lw_plot_from_gui)
        exp_check.grid(row=0, column=8, padx=5, pady=2, sticky='w')
        update_button = Button(control_frame, text="Update Plot", command=_update_lw_plot_from_gui)
        update_button.grid(row=0, column=9, padx=10, pady=2, sticky='e')
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(3, weight=1)
        control_frame.columnconfigure(5, weight=0) 
        control_frame.columnconfigure(7, weight=0)
        control_frame.columnconfigure(8, weight=1) 
        control_frame.columnconfigure(9, weight=0) 
        lw_fig = plt.figure(figsize=(7, 5.5))
        lw_ax = lw_fig.add_subplot(111)
        lw_canvas = FigureCanvasTkAgg(lw_fig, master=lw_window)
        canvas_widget = lw_canvas.get_tk_widget()
        canvas_widget.pack(side='top', fill='both', expand=True)
        toolbar_frame = Frame(lw_window)
        toolbar_frame.pack(side='bottom', fill='x')
        toolbar = NavigationToolbar2Tk(lw_canvas, toolbar_frame)
        toolbar.update()
        _update_lw_plot_from_gui()
        lw_window.lift()
        lw_window.attributes('-topmost', True)
        lw_window.after_idle(lw_window.attributes, '-topmost', False)

    def plot_loomis_wood(target_ax, folding_const, min_intensity, 
                         upper_qn_value_filter, qn_filter_index_0_based,
                         plot_experimental):
        nonlocal entries, loomis_wood_data_store, loomis_wood_data_store_exp, stats_label, dialog, lw_fig, lw_canvas
        if target_ax is None or lw_fig is None or lw_canvas is None:
             print("LW Error: Plotting failed, target axes/figure/canvas invalid.")
             return
        print(f"LW Plot: Fold={folding_const}, MinInt={min_intensity}, QNValFilt={upper_qn_value_filter} on QN Index={qn_filter_index_0_based}, PlotExp={plot_experimental}")
        target_ax.cla()
        loomis_wood_data_store.clear()
        loomis_wood_data_store_exp.clear()
        folding_width = folding_const
        calculated_points_plotted = 0
        experimental_points_plotted = 0
        global STATE_INDEX
        calculated_marker_size = 25
        calculated_marker_transparency = 1.0
        experimental_marker_size = 55
        experimental_marker_transparency = 0.3
        lw_filtered_entries_calc = []
        for i, entry in enumerate(entries):
            if abs(entry['intensity']) < abs(min_intensity): continue
            upper_qns = entry['triple2']
            if upper_qn_value_filter is not None:
                if qn_filter_index_0_based >= 0 and qn_filter_index_0_based < len(upper_qns):
                    if upper_qns[qn_filter_index_0_based] != upper_qn_value_filter:
                        continue
                else:
                    continue
            delta_qn1 = 0
            if STATE_INDEX >= 1:
                try:
                    delta_qn1 = upper_qns[0] - entry['triple1'][0]
                except IndexError:
                     print(f"Warning (LW Calc): Cannot determine branch, QN tuple {upper_qns} or {entry['triple1']} too short.")
                     delta_qn1 = 0 
            branch = 'Q' if delta_qn1 == 0 else ('R' if delta_qn1 == 1 else 'P')
            freq = entry['freq']
            if folding_width <= 0:
                folded_freq, segment_index = freq, 0
            elif freq < 0:
                 folded_freq = (freq % folding_width + folding_width) % folding_width
                 segment_index = np.floor(freq / folding_width)
            else:
                 folded_freq = freq % folding_width
                 segment_index = np.floor(freq / folding_width)
            lw_filtered_entries_calc.append({
                'original_index': i, 'entry_data': entry, 'branch': branch,
                'folded_freq': folded_freq, 'segment_index': segment_index, 'source': 'calc'
            })
        plot_data_calc = {'P': {'x': [], 'y': [], 'data_indices': []},
                          'Q': {'x': [], 'y': [], 'data_indices': []},
                          'R': {'x': [], 'y': [], 'data_indices': []}}
        for idx, data_point in enumerate(lw_filtered_entries_calc):
            branch = data_point['branch']
            if branch in plot_data_calc:
                plot_data_calc[branch]['x'].append(data_point['folded_freq'])
                # Use the original frequency for the Y-axis
                plot_data_calc[branch]['y'].append(data_point['entry_data']['freq'])
                plot_data_calc[branch]['data_indices'].append(idx)
        artists_calc = {}
        colors_calc = {'P': 'blue', 'Q': 'green', 'R': 'red'}
        markers_calc = {'P': 'x', 'Q': 'o', 'R': '+'}
        for branch, data in plot_data_calc.items():
            if data['x']:
                artist = target_ax.scatter(data['x'], data['y'],
                                          label=f'Calc {branch}',
                                          color=colors_calc[branch], marker=markers_calc[branch],
                                          s=calculated_marker_size, picker=5, alpha=calculated_marker_transparency, zorder=5)
                artists_calc[branch] = artist
                loomis_wood_data_store[artist] = {'indices': data['data_indices'], 'source_list': lw_filtered_entries_calc}
                calculated_points_plotted += len(data['x'])
        exp_entries_for_lw = []
        if plot_experimental:
            if not _process_peaklist_for_lw(peaklist_filename, output_auxil_filename='auxil_peaklist_lw.txt'):
                print("LW Warning: Failed to process Peaklist, cannot plot experimental data.")
            else:
                exp_entries_for_lw = _read_auxil_for_lw()
            if exp_entries_for_lw:
                lw_filtered_entries_exp = []
                for i, entry in enumerate(exp_entries_for_lw):
                    if abs(entry['intensity']) < abs(min_intensity): continue
                    upper_qns = entry['triple2']
                    if upper_qn_value_filter is not None:
                        if qn_filter_index_0_based >= 0 and qn_filter_index_0_based < len(upper_qns):
                            if upper_qns[qn_filter_index_0_based] != upper_qn_value_filter:
                                continue
                        else:
                            continue
                    delta_qn1 = 0
                    if STATE_INDEX >= 1:
                         try:
                             delta_qn1 = upper_qns[0] - entry['triple1'][0]
                         except IndexError:
                              print(f"Warning (LW Exp): Cannot determine branch, QN tuple {upper_qns} or {entry['triple1']} too short.")
                              delta_qn1 = 0
                    branch = 'Q' if delta_qn1 == 0 else ('R' if delta_qn1 == 1 else 'P')
                    freq = entry['freq']
                    if folding_width <= 0:
                        folded_freq, segment_index = freq, 0
                    elif freq < 0:
                         folded_freq = (freq % folding_width + folding_width) % folding_width
                         segment_index = np.floor(freq / folding_width)
                    else:
                         folded_freq = freq % folding_width
                         segment_index = np.floor(freq / folding_width)
                    lw_filtered_entries_exp.append({
                        'original_index': i, 'entry_data': entry, 'branch': branch,
                        'folded_freq': folded_freq, 'segment_index': segment_index, 'source': 'exp'
                    })
                plot_data_exp = {'P': {'x': [], 'y': [], 'data_indices': []},
                                 'Q': {'x': [], 'y': [], 'data_indices': []},
                                 'R': {'x': [], 'y': [], 'data_indices': []}}
                for idx, data_point in enumerate(lw_filtered_entries_exp):
                    branch = data_point['branch']
                    if branch in plot_data_exp:
                        plot_data_exp[branch]['x'].append(data_point['folded_freq'])
                        # Use the original frequency for the Y-axis
                        plot_data_exp[branch]['y'].append(data_point['entry_data']['freq'])
                        plot_data_exp[branch]['data_indices'].append(idx)
                artists_exp = {}
                colors_exp = {'P': 'cyan', 'Q': 'lime', 'R': 'fuchsia'}
                markers_exp = {'P': 's', 'Q': 'D', 'R': '^'}
                for branch, data in plot_data_exp.items():
                    if data['x']:
                        artist = target_ax.scatter(data['x'], data['y'],
                                                  label=f'Obs {branch}',
                                                  color=colors_exp[branch], marker=markers_exp[branch],
                                                  s=experimental_marker_size, picker=5, alpha=experimental_marker_transparency, zorder=10,
                                                  edgecolors='black', linewidths=0.5)
                        artists_exp[branch] = artist
                        loomis_wood_data_store_exp[artist] = {'indices': data['data_indices'], 'source_list': lw_filtered_entries_exp}
                        experimental_points_plotted += len(data['x'])
        target_ax.set_xlabel(f'Modulo (transition frequency, folding constant)')
        target_ax.set_ylabel('Transition frequency')
        filter_qn_display_index = qn_filter_index_0_based + 1

        total_points = calculated_points_plotted + experimental_points_plotted
        if total_points == 0:
            msg = f"No data points for LW plot with current filters."
            target_ax.text(0.5, 0.5, msg, horizontalalignment='center', verticalalignment='center', transform=target_ax.transAxes)
            if dialog and dialog.winfo_exists(): stats_label.config(text=msg)
        else:
            # locate legend to the title position ---
            # 'bbox_to_anchor' places it just above the plot area.
            # 'loc=lower center' anchors the legend's bottom-middle to that point.
            # 'ncol' arranges the items horizontally.
            # 'frameon=False' removes the border for a cleaner look.
            target_ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                           ncol=6, fontsize='small', frameon=False)
            target_ax.grid(True, linestyle=':', alpha=0.6)
            status_msg = f"LW Plot: {calculated_points_plotted} Calc pts"
            if plot_experimental:
                status_msg += f", {experimental_points_plotted} Exp pts plotted."
            else:
                status_msg += " plotted."
            if dialog and dialog.winfo_exists(): stats_label.config(text=status_msg)
        canvas_to_check = lw_fig.canvas
        pick_callbacks = canvas_to_check.callbacks.callbacks.get('pick_event', {})
        is_connected = any(cb == on_lw_pick for cb in pick_callbacks.values())
        if not is_connected:
            print("Connecting LW pick event handler.")
            canvas_to_check.mpl_connect('pick_event', on_lw_pick)
        try:
             lw_fig.canvas.draw_idle()
        except Exception as e:
             print(f"Error during LW canvas draw: {e}")

    def on_lw_pick(event):
        nonlocal current_line, stats_label, loomis_wood_data_store, loomis_wood_data_store_exp, dialog, lw_ax, lw_highlight_marker
        global fig, ax1
        if ax1 is None or fig is None:
            print("Main plot (ax1/fig) not available for interaction.")
            return
        artist = event.artist
        ind = event.ind
        if not len(ind): return
        clicked_point_index_in_artist = ind[0]
        data_info = None
        point_source = None
        line_color = 'green'
        if artist in loomis_wood_data_store:
            data_info = loomis_wood_data_store[artist]
            point_source = 'Calc'
            line_color = 'green'
            print(f"LW Pick (Calc): Artist={artist}, Index={clicked_point_index_in_artist}")
        elif artist in loomis_wood_data_store_exp:
            data_info = loomis_wood_data_store_exp[artist]
            point_source = 'Exp'
            line_color = 'red'
            print(f"LW Pick (Exp): Artist={artist}, Index={clicked_point_index_in_artist}")
        else:
             print("LW Pick Error: Picked artist not found in known data stores. Artist:", artist)
             return
        source_list = data_info['source_list']
        indices_in_source = data_info['indices']
        if clicked_point_index_in_artist >= len(indices_in_source):
            print(f"LW Pick Error: Index {clicked_point_index_in_artist} out of bounds for artist's indices (len={len(indices_in_source)}).")
            return
        original_filtered_list_index = indices_in_source[clicked_point_index_in_artist]
        if original_filtered_list_index >= len(source_list):
            print(f"LW Pick Error: Mapped index {original_filtered_list_index} out of bounds for source list (len={len(source_list)}).")
            return
        selected_data_point = source_list[original_filtered_list_index]
        selected_data = selected_data_point['entry_data']
        try:
            freq = selected_data['freq']
            intensity = selected_data['intensity']
            lower_qns = selected_data['triple1']
            upper_qns = selected_data['triple2']
        except KeyError as e:
            print(f"LW Pick Error: Missing key {e} in selected_data: {selected_data}")
            if dialog and dialog.winfo_exists(): stats_label.config(text=f"LW Pick Error: Data structure issue.")
            return
        except Exception as e:
             print(f"LW Pick Error: Error extracting data: {e}")
             if dialog and dialog.winfo_exists(): stats_label.config(text=f"LW Pick Error: Cannot read data.")
             return
        
        # --- Logic to remove the old highlight MARKER ---
        if lw_highlight_marker is not None:
            try:
                lw_highlight_marker.remove()
            except Exception as e_rem:
                print(f"Warning: Could not remove previous LW highlight marker: {e_rem}")
        lw_highlight_marker = None # Always reset the reference
        # --- END Logic ---

        _write_selection_to_labels_temp(lower_qns, upper_qns)
        lower_qns_display = " ".join(map(str, lower_qns))
        upper_qns_display = " ".join(map(str, upper_qns))
        status_text = f"LW Pick ({point_source}): F={freq:.6f}, I={intensity:.3e} | {lower_qns_display} → {upper_qns_display}"
        try:
            if dialog and dialog.winfo_exists():
                 stats_label.config(text=status_text)
            else:
                 print("LW Pick Info: Main dialog closed, cannot update status label.")
        except tk.TclError as e:
             print(f"LW Pick Warning: Failed to update stats_label (likely closed): {e}.")
        except Exception as e:
            print(f"LW Pick Error: Unexpected error updating stats_label: {e}")
        try:
            if input_entry and input_entry.winfo_exists():
                upper_qns_str = ' '.join(map(str, upper_qns))
                input_entry.delete(0, 'end')
                input_entry.insert(0, upper_qns_str)
                perform_search()
            else:
                print("LW Pick Info: Main input_entry widget closed or unavailable.")
        except Exception as e:
            print(f"LW Pick Error: Failed to update input_entry widget: {e}")
        if current_line is not None:
            try:
                if current_line in ax1.lines: current_line.remove()
            except Exception as e: print(f"Warning: Error removing previous line from main plot: {e}")
            current_line = None

        # --- Logic to draw the new highlight MARKER using scatter ---
        try:
            if lw_ax and lw_fig and lw_fig.canvas:
                folded_freq = selected_data_point['folded_freq']
                selected_wavenumber = selected_data_point['entry_data']['freq']

                # Use scatter to draw a single, larger, semi-transparent point
                lw_highlight_marker = lw_ax.scatter(
                    [folded_freq],  # x must be in a list
                    [selected_wavenumber],# y must be in a list
                    s=200,          # Size of the marker in points^2 (e.g., 200 is quite large)
                    facecolors='yellow', # The fill color
                    edgecolors='red',    # The border color of the marker
                    alpha=0.45,      # Semi-transparent
                    linewidths=1.5,
                    zorder=1
                )
                
                # Redraw the Loomis-Wood canvas to show the new marker
                lw_fig.canvas.draw_idle()
        except Exception as e_marker:
            print(f"Error creating LW highlight marker: {e_marker}")
        # --- END Logic ---

        try:
            y_lim_main = ax1.get_ylim()
            plot_intensity_main = abs(intensity) if intensity != 0 else 1e-9
            plot_y_end_main = min(plot_intensity_main, y_lim_main[1])
            plot_y_start_main = max(0.0, y_lim_main[0])
            if y_lim_main[1] <= y_lim_main[0]:
                 plot_y_start_main = 0; plot_y_end_main = plot_intensity_main
            elif plot_y_end_main <= y_lim_main[0]:
                 plot_y_start_main = y_lim_main[0]
                 plot_y_end_main = y_lim_main[0] + 0.05 * (y_lim_main[1] - y_lim_main[0])
            elif plot_y_start_main >= plot_y_end_main:
                 plot_y_start_main = max(0.0, y_lim_main[0])
                 plot_y_end_main = plot_y_start_main + 0.05 * max(1e-9, (y_lim_main[1] - y_lim_main[0]))
            current_line = ax1.vlines(freq, plot_y_start_main, plot_y_end_main,
                                      colors=line_color, linestyles='-', linewidth=1.5,
                                      label=f'LW Sel ({point_source})', zorder=11)
            current_xlim_main = ax1.get_xlim()
            if current_xlim_main[1] > current_xlim_main[0]: x_range_main = current_xlim_main[1] - current_xlim_main[0]
            else: x_range_main = freq * 0.1 if freq > 0 else 10.0
            new_xlim_main = (freq - x_range_main / 2, freq + x_range_main / 2)
            min_x_range = 1e-5
            if not all(np.isfinite(x) for x in new_xlim_main) or abs(new_xlim_main[1] - new_xlim_main[0]) < min_x_range:
                 new_xlim_main = (freq - min_x_range*5, freq + min_x_range*5)
            ax1.set_xlim(new_xlim_main)
            fig.canvas.draw_idle()
            print(f"Main plot updated for Freq={freq} from LW pick ({point_source}).")
        except Exception as e:
            print(f"Error during main plot interaction from LW pick: {e}")
            if dialog and dialog.winfo_exists():
                stats_label.config(text=f"{status_text} | Error updating main plot: {e}")
            messagebox.showerror("Plot Error", f"Error updating main plot from LW pick: {e}", parent=dialog)
            current_line = None

    def _load_assignments_from_database(filepath):
        """
        Parses the Peaklist database and returns a dictionary mapping an assigned
        transition to its experimental peak's frequency and intensity.

        Args:
            filepath (str): The path to the Peaklist.txt file.

        Returns:
            dict: A map where keys are (lower_qns_tuple, upper_qns_tuple) and
                values are (exp_freq, exp_intensity).
        """
        global STATE_INDEX
        assignment_map = {}
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    
                    try:
                        exp_freq = float(parts[1])
                        exp_intensity = float(parts[2])
                        contour_type = parts[3]
                        
                        if contour_type == 'Voigt':
                            assignment_start_index = 7
                        else: # Gaussian or Lorentzian
                            assignment_start_index = 6
                        
                        if len(parts) < assignment_start_index + (2 * STATE_INDEX):
                            continue # No full assignment block
                        
                        assignment_parts_str = parts[assignment_start_index:]
                        block_size = 2 * STATE_INDEX
                        
                        for i in range(0, len(assignment_parts_str), block_size):
                            block_str = assignment_parts_str[i : i + block_size]
                            if len(block_str) == block_size:
                                block_int = list(map(int, block_str))
                                lower_qns = tuple(block_int[0 : STATE_INDEX])
                                upper_qns = tuple(block_int[STATE_INDEX : block_size])
                                transition_key = (lower_qns, upper_qns)
                                # If a transition is assigned to multiple peaks, this will
                                # keep the data from the last one found in the file.
                                assignment_map[transition_key] = (exp_freq, exp_intensity)
                    except (ValueError, IndexError):
                        # Gracefully skip malformed lines
                        continue
        except FileNotFoundError:
            print(f"Warning (load_assignments): Database file {filepath} not found for cross-referencing.")
        
        return assignment_map

# --- [* NEW *] ---
    # Functionality for the new "Summary" button.
    def show_summary():
        """Creates and displays the summary window with advanced, stateful filters
        for quantum numbers."""
        
        summary_window = tk.Toplevel(dialog)
        summary_window.title("Summary")
        position_dialog_near_cursor(summary_window, width=850, height=600)

        # --- State variables for the new filter controls ---
        filter_values = {}  # Stores {filter_index (1-based): filter_value}
        current_filter_index = 1

        # --- Top frame for all the controls ---
        control_frame = tk.Frame(summary_window, pady=5)
        control_frame.pack(side='top', fill='x', padx=10)

        # --- Main area which will contain the scrollable canvas ---
        main_area = tk.Frame(summary_window)
        main_area.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        # --- Functions for the new filter controls ---
        def save_current_filter():
            """Reads the filter entry, validates, and saves it to the dictionary."""
            value_str = filter_value_entry.get().strip()
            if value_str:
                try:
                    filter_values[current_filter_index] = int(value_str)
                except ValueError:
                    messagebox.showwarning("Invalid Filter", f"Filter value '{value_str}' is not a valid integer. It will be ignored.", parent=summary_window)
                    # If invalid, remove any existing filter for this index
                    if current_filter_index in filter_values:
                        del filter_values[current_filter_index]
            # If the entry is empty, remove any existing filter for this index
            elif current_filter_index in filter_values:
                del filter_values[current_filter_index]
        
        def update_filter_ui():
            """Updates the filter index label and entry box."""
            filter_index_label.config(text=str(current_filter_index))
            # Load the stored value for the new index, or show an empty string
            new_value = filter_values.get(current_filter_index, "")
            filter_value_entry.delete(0, 'end')
            filter_value_entry.insert(0, str(new_value))

        def increment_filter_index():
            """Saves current filter, increments index, and updates UI."""
            nonlocal current_filter_index
            save_current_filter()
            if current_filter_index < STATE_INDEX:
                current_filter_index += 1
                update_filter_ui()
        
        def decrement_filter_index():
            """Saves current filter, decrements index, and updates UI."""
            nonlocal current_filter_index
            save_current_filter()
            if current_filter_index > 1:
                current_filter_index -= 1
                update_filter_ui()
        
        # =====================================================================
        # --- NEW FUNCTION TO HANDLE CLICKS ON THE RESULTS TABLE ---
        # =====================================================================
        def on_table_row_click(transition_data):
            """
            Handles a click on a row in the results table. Centers the main
            plot on the transition and updates LabelsTemp.txt.
            """
            try:
                freq = transition_data['freq']
                lower_qns = transition_data['lower_qns']
                upper_qns = transition_data['upper_qns']
            except (KeyError, TypeError):
                print("Error: Could not extract data from clicked table row.")
                return

            # Action 1: Write to LabelsTemp.txt
            _write_selection_to_labels_temp(lower_qns, upper_qns)
            print(f"Wrote QNs to LabelsTemp.txt: {lower_qns} -> {upper_qns}")

            # Action 2: Center the main plot
            if fig is None or ax1 is None:
                print("Warning: Main plot not available for interaction.")
                return
            
            try:
                current_xlim = ax1.get_xlim()
                x_range = current_xlim[1] - current_xlim[0]
                # Ensure a valid range even if plot is not properly zoomed
                if x_range <= 0 or not np.isfinite(x_range):
                    x_range = 1.0 
                
                new_xlim = (freq - x_range / 2, freq + x_range / 2)
                
                # Set limits on both main axes for synchronization
                ax1.set_xlim(new_xlim)
                ax2.set_xlim(new_xlim)
                
                fig.canvas.draw_idle()
                print(f"Main plot centered on calculated frequency: {freq:.6f}")

            except Exception as e:
                print(f"Error centering main plot: {e}")
        
        # =====================================================================
        # --- FUNCTION TO BE CALLED WHEN A GRID CELL IS CLICKED ---
        # =====================================================================
        def on_cell_click(x_val, y_val, x_idx, y_idx, current_filters):
            """
            Handles the logic when a user clicks a cell in the summary grid.
            Filters the calculated linelist, cross-references with the database,
            and displays the results in a new table window.
            """
            # --- 1. Construct the complete filter for the upper state ---
            final_upper_state_filter = current_filters.copy()
            final_upper_state_filter[x_idx + 1] = int(x_val) 
            final_upper_state_filter[y_idx + 1] = int(y_val) 
            
            # --- 2. Load the calculated linelist ---
            all_calc_transitions = []
            try:
                with open(calculated_file, 'r') as f_calc:
                    for line in f_calc:
                        parts = line.strip().split()
                        if len(parts) < 2 + (2 * STATE_INDEX): continue
                        try:
                            all_calc_transitions.append({
                                'freq': float(parts[0]),
                                'intensity': float(parts[1]),
                                'lower_qns': tuple(map(int, parts[2 : 2+STATE_INDEX])),
                                'upper_qns': tuple(map(int, parts[2+STATE_INDEX : 2+(2*STATE_INDEX)]))
                            })
                        except (ValueError, IndexError):
                            continue
            except FileNotFoundError:
                messagebox.showerror("File Error", f"Calculated linelist file not found:\n{calculated_file}", parent=summary_window)
                return

            # --- 3. Filter the calculated transitions ---
            matching_transitions = []
            for trans in all_calc_transitions:
                upper_qns = trans['upper_qns']
                is_match = True
                for qn_idx_1based, qn_val in final_upper_state_filter.items():
                    if upper_qns[qn_idx_1based - 1] != qn_val:
                        is_match = False
                        break
                if is_match:
                    matching_transitions.append(trans)
            
            # --- 4. Sort by descending intensity ---
            matching_transitions.sort(key=lambda t: t['intensity'], reverse=True)
            
            if not matching_transitions:
                messagebox.showinfo("No Results", "No calculated transitions found for the selected quantum state.", parent=summary_window)
                return

            # --- 5. Cross-reference with the database ---
            assignment_map = _load_assignments_from_database(peaklist_filename)
            
            # --- 6. Show the results window ---
            results_window = tk.Toplevel(summary_window)
            results_window.title(f"Transitions for Upper State")
            position_dialog_near_cursor(results_window, 600, 500)
            
            res_canvas = tk.Canvas(results_window); res_scrollbar = tk.Scrollbar(results_window, orient="vertical", command=res_canvas.yview); res_table_frame = tk.Frame(res_canvas)
            res_canvas.configure(yscrollcommand=res_scrollbar.set); res_scrollbar.pack(side="right", fill="y"); res_canvas.pack(side="left", fill="both", expand=True)
            res_canvas.create_window((0, 0), window=res_table_frame, anchor="nw")
            res_table_frame.bind("<Configure>", lambda e: res_canvas.configure(scrollregion=res_canvas.bbox("all")))
            
            headers = ["TRANSITION LABELS", "FREQUENCY", "INTENSITY", "DB"]
            for col_idx, header_text in enumerate(headers):
                header = tk.Label(res_table_frame, text=header_text, font=("Arial", 10, "bold"), bd=1, relief='raised', padx=5, pady=2)
                header.grid(row=0, column=col_idx, sticky="nsew")
            res_table_frame.columnconfigure(0, weight=3)

            # Populate Table
            for row_idx, trans in enumerate(matching_transitions, start=1):
                transition_key = (trans['lower_qns'], trans['upper_qns'])
                
                # ALWAYS use the calculated frequency and intensity
                freq, intensity = trans['freq'], trans['intensity']
                
                # ONLY use the database map to determine the DB marker
                db_marker = "+" if transition_key in assignment_map else ""
                
                lower_str = " ".join(map(str, trans['lower_qns']))
                upper_str = " ".join(map(str, trans['upper_qns']))
                label_str = f"{lower_str}  -->  {upper_str}"

                row_data = [label_str, f"{freq:.6f}", f"{intensity:.4e}", db_marker]
                bg_color = "#f0f0f0" if row_idx % 2 == 0 else "white"
                
                # --- MODIFICATION: Make the entire row clickable ---
                for col_idx, cell_text in enumerate(row_data):
                    cell = tk.Label(res_table_frame, text=cell_text, anchor='w', padx=5, pady=2, bg=bg_color)
                    cell.grid(row=row_idx, column=col_idx, sticky="nsew")
                    # Set the cursor to a hand to indicate interactivity
                    cell.config(cursor="hand2")
                    # Bind the click event, passing the full transition data object
                    cell.bind("<Button-1>", lambda event, t=trans: on_table_row_click(t))
                # --- END MODIFICATION ---
        
        # --- Command for the 'show' button ---
        def on_show_click():
            """Parses inputs, processes Peaklist with filters, and draws the grid."""
            
            # Save the currently displayed filter value before processing
            save_current_filter()

            for widget in main_area.winfo_children():
                widget.destroy()

            try:
                # Helper functions from previous steps...
                def parse_lim_values(lim_str): # (omitting full code for brevity, it's the same)
                    if not lim_str.strip(): return []
                    tokens, expanded = [t for t in lim_str.replace(",", " ").split(" ") if t], []
                    for token in tokens:
                        if "-" in token:
                            parts = token.split("-"); start, end = int(parts[0]), int(parts[1])
                            expanded.extend(list(range(start, end + 1)))
                        else: expanded.append(float(token))
                    return sorted(list(set(expanded)))
                def parse_cell_size(size_str): # (omitting full code for brevity, it's the same)
                    if not size_str.strip(): return 10, 2
                    parts = [p for p in size_str.replace(",", " ").split(" ") if p]
                    if len(parts) != 2: raise ValueError("Cell size must have two numbers")
                    return int(parts[0]), int(parts[1])

                x_qn_index = int(x_label_entry.get().strip()) - 1
                y_qn_index = int(y_label_entry.get().strip()) - 1
                x_lim_values = parse_lim_values(x_lim_entry.get())
                y_lim_values = parse_lim_values(y_lim_entry.get())
                cell_width, cell_height = parse_cell_size(cell_size_entry.get())

                if not x_lim_values or not y_lim_values: raise ValueError("'x lim' and 'y lim' must not be empty")
                if not (0 <= x_qn_index < STATE_INDEX and 0 <= y_qn_index < STATE_INDEX):
                    raise ValueError(f"Label numbers must be between 1 and {STATE_INDEX}")
            
            except ValueError as e:
                messagebox.showerror("Input Error", f"Invalid input: {e}", parent=summary_window)
                return
            
            counts_data = {}


            try:
                block_size = 2 * STATE_INDEX
                with open(peaklist_filename, "r") as f:
                    for line in f:
                        parts = line.strip().split()

                        # Determine start index and perform robust checks
                        if len(parts) < 4: continue # Not even enough for a basic peak
                        contour_type = parts[3]
                        
                        if contour_type == 'Voigt':
                            assignment_start_index = 7
                        else: # Gaussian or Lorentzian
                            assignment_start_index = 6
                        
                        # Now, check the length with the correct start index
                        if len(parts) < assignment_start_index + block_size:
                            continue
                        
                        assignment_parts_str = parts[assignment_start_index:]
                        # --- END FIX ---
                        
                        for i in range(0, len(assignment_parts_str), block_size):
                            block_str = assignment_parts_str[i : i + block_size]
                            if len(block_str) < block_size: continue

                            try:
                                upper_state_qns_str = block_str[STATE_INDEX:]
                                
                                # --- NEW FILTERING LOGIC ---
                                passes_all_filters = True
                                for f_idx, f_val in filter_values.items():
                                    # f_idx is 1-based, list index is 0-based
                                    qn_to_check = int(upper_state_qns_str[f_idx - 1])
                                    if qn_to_check != f_val:
                                        passes_all_filters = False
                                        break # No need to check other filters for this block
                                
                                if not passes_all_filters:
                                    continue # Skip to the next assignment block
                                # --- END OF FILTERING LOGIC ---

                                x_qn_val = int(upper_state_qns_str[x_qn_index])
                                y_qn_val = int(upper_state_qns_str[y_qn_index])
                                key = (x_qn_val, y_qn_val)
                                counts_data[key] = counts_data.get(key, 0) + 1
                            except (ValueError, IndexError):
                                continue
            except FileNotFoundError:
                print("Info (Summary): Peaklist not found.")
            except Exception as e:
                messagebox.showerror("File Error", f"Error reading Peaklist: {e}", parent=summary_window)
                return

            # UI Rendering (same as before, but the data is now filtered)
            canvas = tk.Canvas(main_area); y_scrollbar = tk.Scrollbar(main_area, orient='vertical', command=canvas.yview); x_scrollbar = tk.Scrollbar(main_area, orient='horizontal', command=canvas.xview); grid_frame = tk.Frame(canvas)
            canvas.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set); y_scrollbar.pack(side='right', fill='y'); x_scrollbar.pack(side='bottom', fill='x'); canvas.pack(side='left', fill='both', expand=True); canvas.create_window((0, 0), window=grid_frame, anchor="nw")
            def on_frame_configure(event): canvas.configure(scrollregion=canvas.bbox("all"))
            grid_frame.bind("<Configure>", on_frame_configure)
            for j, x_val in enumerate(x_lim_values): header = tk.Label(grid_frame, text=f"{x_val:g}", font=("Arial", 10, "bold")); header.grid(row=0, column=j + 1, sticky="nsew")
            for i, y_val in enumerate(y_lim_values):
                header = tk.Label(grid_frame, text=f"{y_val:g}", font=("Arial", 10, "bold")); header.grid(row=i + 1, column=0, sticky="nsew")
                for j, x_val in enumerate(x_lim_values):
                    base_text = f"{x_val:g},{y_val:g}"
                    count = counts_data.get((x_val, y_val), 0)
                    
                    # --- MODIFICATION: Make cells clickable ---
                    if count > 0:
                        cell = tk.Label(grid_frame, text=f"{base_text}\n({count})", relief="solid", borderwidth=1, width=cell_width, height=cell_height, fg="green", font=("Arial", 9, "bold"))
                        # Change cursor to indicate interactivity
                        cell.config(cursor="hand2")
                    else:
                        cell = tk.Label(grid_frame, text=base_text, relief="solid", borderwidth=1, width=cell_width, height=cell_height)
                    
                    cell.grid(row=i + 1, column=j + 1, sticky="nsew")
                    # Bind the click event, using a lambda to pass the cell's context
                    cell.bind("<Button-1>", lambda event, xv=x_val, yv=y_val: on_cell_click(xv, yv, x_qn_index, y_qn_index, filter_values))
                    # --- END MODIFICATION ---

        # --- Create and pack the control widgets ---
        # Using grid for precise control
        control_frame.columnconfigure(11, weight=1) # Make space before button expand

        tk.Label(control_frame, text="horizontal label #").grid(row=0, column=0); x_label_entry = tk.Entry(control_frame, width=5); x_label_entry.grid(row=0, column=1, padx=(0, 5))
        tk.Label(control_frame, text="horizontal limits").grid(row=0, column=2); x_lim_entry = tk.Entry(control_frame, width=12); x_lim_entry.grid(row=0, column=3, padx=(0, 5))
        tk.Label(control_frame, text="vertical label #").grid(row=0, column=4); y_label_entry = tk.Entry(control_frame, width=5); y_label_entry.grid(row=0, column=5, padx=(0, 5))
        tk.Label(control_frame, text="vertical limits").grid(row=0, column=6); y_lim_entry = tk.Entry(control_frame, width=12); y_lim_entry.grid(row=0, column=7, padx=(0, 5))
        tk.Label(control_frame, text="cell size").grid(row=0, column=8); cell_size_entry = tk.Entry(control_frame, width=8); cell_size_entry.insert(0, "10, 2"); cell_size_entry.grid(row=0, column=9, padx=(0, 5))

        # --- New Filter Controls ---
        filter_frame = tk.Frame(control_frame); filter_frame.grid(row=0, column=10, padx=(10, 0))
        tk.Label(filter_frame, text="filter label").pack(side='left')
        tk.Button(filter_frame, text=u"\u25B2", command=increment_filter_index, font=("Arial", 6)).pack(side='left') # Up arrow
        tk.Button(filter_frame, text=u"\u25BC", command=decrement_filter_index, font=("Arial", 6)).pack(side='left') # Down arrow
        filter_index_label = tk.Label(filter_frame, text="1", width=2, relief="sunken"); filter_index_label.pack(side='left')
        filter_value_entry = tk.Entry(filter_frame, width=4); filter_value_entry.pack(side='left')

        show_button = tk.Button(control_frame, text="show", command=on_show_click); show_button.grid(row=0, column=12, sticky='e')

        summary_window.lift(); summary_window.attributes('-topmost', True); summary_window.after_idle(summary_window.attributes, '-topmost', False)


    # --- Buttons (Main Dialog) ---
    button_frame = Frame(dialog) 
    button_frame.pack(pady=5, padx=10, fill='x')
    Button(button_frame, text="Search", command=perform_search).pack(side='left', padx=3)
    Button(button_frame, text="Evaluate", command=perform_evaluate).pack(side='left', padx=3)
    Button(button_frame, text="Calculation DB", command=show_summary).pack(side='left', padx=3)
    Button(button_frame, text="Loomis-Wood", command=create_or_update_loomis_wood_window).pack(side='left', padx=3)
    Button(button_frame, text="Make output", command=lambda: Process_data('Gen_output.txt')).pack(side='left', padx=3)

    # --- Listbox (Main Dialog) ---
    table_frame = Frame(dialog)
    table_frame.pack(fill='both', expand=True, padx=10, pady=5)
    scrollbar = Scrollbar(table_frame)
    scrollbar.pack(side='right', fill='y')
    listbox = Listbox(table_frame, yscrollcommand=scrollbar.set, width=100, height=15)
    listbox.pack(fill='both', expand=True)
    scrollbar.config(command=listbox.yview)
    listbox.bind('<<ListboxSelect>>', on_select)


    # --- Dialog Close Handling ---
    def on_dialog_close():
        nonlocal lw_window, tk_root_required, parent_window
        print("Closing Interpretation dialog.")
        if lw_window and lw_window.winfo_exists():
             print("Closing associated Loomis-Wood window.")
             _on_lw_window_close() 
        # Removed closing of the statistics plots as they are no longer created
        # try: plt.close("Observed-Calculated Frequency Differences")
        # except: pass 
        # try: plt.close("Experimental Energy Standard Deviation")
        # except: pass
        dialog.destroy()
        if tk_root_required and parent_window:
             print("Destroying standalone Tk root.")
             try:
                 parent_window.destroy()
             except tk.TclError:
                 print("Warning: Error destroying standalone Tk root (might already be destroyed).")

    dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)
    dialog.lift()
    dialog.attributes('-topmost', True)
    dialog.after_idle(dialog.attributes, '-topmost', False)

# --- End of Interpretation Function ---

def simple_rovibrational_auto(spectrum_file, calculated_file, lower_states_file, assignment_window, fitting_window, comb_dif_thresh):
    """
    Preliminary implementation of the simple calculation-assisted rovibrational analysis algorithm.
    This function loads and parses the necessary data files, sorts the calculated linelist
    by intensity, and sets up the main processing loop.
    
    Args:
        spectrum_file (str): Path to the main experimental spectrum file.
        calculated_file (str): Path to the calculated linelist file.
        lower_states_file (str): Path to the file with lower state energy levels.
        assignment_window (float): The tolerance for making assignments.
        fitting_window (float): The width of the local region for deconvolution and fitting.
        comb_dif_thresh (float): The specific tolerance for validating a combination difference.
    """
    global STATE_INDEX, peaklist_filename, x, y, MIN_AMP, NUM_STARTS, DEFAULT_CONTOUR, DEFAULT_FWHM, DEFAULT_FWHM_G, DEFAULT_FWHM_L


    # Write starting log message to Gen_Output.txt ---
    with open("Gen_Output.txt", "a") as f_log:
        f_log.write("\n# --- Starting Simple Calculation-Assisted Rovibrational Analysis ---\n")

    # --- Helper function to write the in-memory database back to the file ---
    def _write_database_to_file(db_data, filepath):
        lines_to_write = []
        for peak in db_data:
            parts = [peak['symbol'], f"{peak['center']:.6f}", f"{peak['amplitude']:.6e}", peak['contour']]
            if peak['contour'] == 'Voigt':
                parts.extend([f"{p:.4e}" for p in peak['fwhm_params']])
            else:
                parts.append(f"{peak['fwhm_params'][0]:.4e}")
            parts.append(f"{peak['integral']:.6e}")
            
            for assignment in peak.get('assignments', []):
                parts.extend(map(str, assignment['lower_qns']))
                parts.extend(map(str, assignment['upper_qns']))
            lines_to_write.append(" ".join(parts) + "\n")
        
        safe_write_to_file(filepath, lines_to_write)
        print(f"Safely wrote {len(lines_to_write)} entries to {filepath}")

    # Helper function to load and parse the database file ---
    def _load_database_file(filepath):
        """Reads and parses the peaklist file into a list of dictionaries."""
        db_data = []
        try:
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        parts = line.strip().split()
                        if not parts: continue
                        entry = {}
                        entry['symbol'] = parts[0]
                        entry['center'] = float(parts[1])
                        entry['amplitude'] = float(parts[2])
                        entry['contour'] = parts[3]
                        assignment_start_index = 0
                        if entry['contour'] == 'Voigt':
                            if len(parts) < 7: continue
                            entry['fwhm_params'] = (float(parts[4]), float(parts[5]))
                            entry['integral'] = float(parts[6])
                            assignment_start_index = 7
                        else:
                            if len(parts) < 6: continue
                            entry['fwhm_params'] = (float(parts[4]),)
                            entry['integral'] = float(parts[5])
                            assignment_start_index = 6
                        entry['assignments'] = []
                        assignment_parts = parts[assignment_start_index:]
                        block_size = 2 * STATE_INDEX
                        if len(assignment_parts) % block_size == 0:
                            for i in range(0, len(assignment_parts), block_size):
                                block = assignment_parts[i : i + block_size]
                                lower_qns = tuple(map(int, block[0 : STATE_INDEX]))
                                upper_qns = tuple(map(int, block[STATE_INDEX : block_size]))
                                entry['assignments'].append({'lower_qns': lower_qns, 'upper_qns': upper_qns})
                        db_data.append(entry)
                    except (ValueError, IndexError):
                        continue
            print(f"Successfully loaded {len(db_data)} peaks from database file: {filepath}")
        except FileNotFoundError:
            print(f"Info: Database file '{filepath}' not found. Starting with an empty database.")
        return db_data
    # --- END

    # --- Helper function to load the lower state energies ---
    def _load_lower_state_energies(filepath):
        energies = {}
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 1 + STATE_INDEX: continue
                    try:
                        energy = float(parts[0])
                        qns = tuple(map(int, parts[1:1+STATE_INDEX]))
                        energies[qns] = energy
                    except (ValueError, IndexError):
                        continue
            print(f"Loaded {len(energies)} lower state energy levels.")
            return energies
        except FileNotFoundError:
            messagebox.showerror("File Not Found", f"The lower states file could not be found:\n{filepath}")
            return None

    def _deconvolve_and_fit_region(center_freq, db_data, exp_x, exp_y_master, fitting_w, data_w):
        """
        [REVISED ALGORITHM] Performs iterative fitting on a spectral region by first seeding
        a dense grid of new peaks and then repeatedly calling the main fitting engine.

        This method replaces the previous residual-based peak finding with a brute-force
        "seed and refine" strategy.

        Args:
            center_freq (float): The center of the region to analyze.
            db_data (list): The current in-memory database of peak dictionaries.
            exp_x (np.ndarray): The x-axis of the experimental spectrum.
            exp_y_master (np.ndarray): The y-axis of the original experimental spectrum.
            fitting_w (float): The width of the window for fitting and peak seeding.
            data_w (float): The width of the wider context window for the simulation.

        Returns:
            list: The updated in-memory database after 5 fitting iterations.
        """
        print(f"  > Entering REMADE deconvolution engine for region ~{center_freq:.4f}...")

        # 1. Define the fitting and simulation windows based on the input parameters.
        window_limits = (center_freq - fitting_w / 2, center_freq + fitting_w / 2)
        sim_window_limits = (center_freq - data_w / 2, center_freq + data_w / 2)

        # 2. Make a working copy of the database to augment with new peaks.
        current_db_data = [p.copy() for p in db_data]

        # --- NEW ALGORITHM: Dense Grid Seeding ---
        # 3. Create a regular grid of X-points with a step of 0.001 inside the fitting window.
        grid_x = np.arange(window_limits[0], window_limits[1], 0.001)

        if grid_x.size > 0:
            # 4. Interpolate the experimental spectrum to get Y-values at the new grid points.
            grid_y = np.interp(grid_x, exp_x, exp_y_master)

            # 5. Add a new 'S' peak to the database for each point on the grid.
            for i in range(len(grid_x)):
                new_center = grid_x[i]
                # Amplitude is set to 70% of the experimental intensity at that point.
                new_amplitude = 0.7 * grid_y[i]

                # Skip adding peaks with negligible or negative amplitude.
                if new_amplitude < 1e-9:
                    continue

                new_peak = {
                    'symbol': 'S',
                    'center': new_center,
                    'amplitude': new_amplitude,
                    'contour': DEFAULT_CONTOUR,
                    'fwhm_params': (DEFAULT_FWHM_G, DEFAULT_FWHM_L) if DEFAULT_CONTOUR == 'Voigt' else (DEFAULT_FWHM,),
                    'integral': 0.0, # Will be calculated by fit_peaks
                    'assignments': []
                }
                current_db_data.append(new_peak)
            
            print(f"    - Seeded {len(grid_x)} new peaks into the database for the fitting region.")
        else:
            print("    - Warning: Fitting window is too small to seed any new peaks.")


        # --- Iterative Fitting Loop ---
        # 6. Prepare the necessary limit files for the main fitter.
        with open("fitlim.txt", "w") as f:
            f.write(f"{window_limits[0]}\n{window_limits[1]}\n")
        
        simlim_lines = [f"{sim_window_limits[0]}\n", f"{sim_window_limits[1]}\n"]
        safe_write_to_file("simlim.txt", simlim_lines)

        # 7. Write the entire augmented database (original + seeded peaks) to the main file.
        # This becomes the starting point for the first fitting iteration.
        _write_database_to_file(current_db_data, peaklist_filename)

        # 8. Call the main fit_peaks function in a loop for exactly 5 iterations.
        for iteration in range(5):
            print(f"    - Starting fitting iteration #{iteration + 1}/5...")
            
            # fit_peaks reads from and writes to peaklist_filename, refining the results on disk.
            fit_peaks(filename, "fitlim.txt", peaklist_filename, MIN_AMP, NUM_STARTS, 1.0e-8)
            
            print(f"    - Fitting iteration #{iteration + 1} complete.")

        # 9. After the loop, load the final, refined results from the file.
        print("  - Deconvolution and fitting cycles complete.")
        final_db_data = _load_database_file(peaklist_filename)

        return final_db_data

    def _find_partner_transition(main_transition, full_calc_linelist, known_lower_energies):
        """
        Searches for the best 'partner' transition for combination differences.
        """
        target_upper_qns = main_transition['upper_qns']
        main_lower_qns = main_transition['lower_qns']
        
        for partner_candidate in full_calc_linelist:
            # A partner must end in the same upper state
            if partner_candidate['upper_qns'] != target_upper_qns:
                continue
            # It must come from a different lower state
            if partner_candidate['lower_qns'] == main_lower_qns:
                continue
            # We must know the energy of the partner's lower state
            if partner_candidate['lower_qns'] not in known_lower_energies:
                continue
            
            # This is a valid partner
            return partner_candidate
            
        return None # No suitable partner found

    # --- 1. Load Data (Calculated Linelist, Database, Lower States) ---

    print(f"--- Starting Automatic Analysis ---")
    print(f"Reading calculated linelist from: {calculated_file}")
    calculated_linelist_data = []
    try:
        with open(calculated_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    parts = line.strip().split()
                    if not parts: continue
                    entry = {}
                    entry['center'] = float(parts[0])
                    entry['intensity'] = float(parts[1])
                    entry['contour'] = parts[2]
                    assignment_start_index = 0
                    if entry['contour'] == 'Voigt':
                        if len(parts) < 5: continue
                        entry['fwhm_params'] = (float(parts[3]), float(parts[4]))
                        assignment_start_index = 5
                    elif entry['contour'] in ['Gaussian', 'Lorentzian']:
                        if len(parts) < 4: continue
                        entry['fwhm_params'] = (float(parts[3]),)
                        assignment_start_index = 4
                    else: continue
                    num_qns = 2 * STATE_INDEX
                    if len(parts) < assignment_start_index + num_qns: continue
                    entry['lower_qns'] = tuple(map(int, parts[assignment_start_index : assignment_start_index + STATE_INDEX]))
                    entry['upper_qns'] = tuple(map(int, parts[assignment_start_index + STATE_INDEX : assignment_start_index + num_qns]))
                    calculated_linelist_data.append(entry)
                except (ValueError, IndexError): continue
        print(f"Successfully loaded {len(calculated_linelist_data)} entries from calculated linelist.")
    except FileNotFoundError:
        messagebox.showerror("File Not Found", f"The calculated linelist file could not be found:\n{calculated_file}")
        return

    print(f"Reading existing database from: {peaklist_filename}")
    database_data = []
    try:
        with open(peaklist_filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    parts = line.strip().split()
                    if not parts: continue
                    entry = {}
                    entry['symbol'] = parts[0]
                    entry['center'] = float(parts[1])
                    entry['amplitude'] = float(parts[2])
                    entry['contour'] = parts[3]
                    assignment_start_index = 0
                    if entry['contour'] == 'Voigt':
                        if len(parts) < 7: continue
                        entry['fwhm_params'] = (float(parts[4]), float(parts[5]))
                        entry['integral'] = float(parts[6])
                        assignment_start_index = 7
                    else:
                        if len(parts) < 6: continue
                        entry['fwhm_params'] = (float(parts[4]),)
                        entry['integral'] = float(parts[5])
                        assignment_start_index = 6
                    entry['assignments'] = []
                    assignment_parts = parts[assignment_start_index:]
                    block_size = 2 * STATE_INDEX
                    if len(assignment_parts) % block_size == 0:
                        for i in range(0, len(assignment_parts), block_size):
                            block = assignment_parts[i : i + block_size]
                            lower_qns = tuple(map(int, block[0 : STATE_INDEX]))
                            upper_qns = tuple(map(int, block[STATE_INDEX : block_size]))
                            entry['assignments'].append({'lower_qns': lower_qns, 'upper_qns': upper_qns})
                    database_data.append(entry)
                except (ValueError, IndexError): continue
        print(f"Successfully loaded {len(database_data)} peaks from database file.")
    except FileNotFoundError:
        print(f"Info: Database file '{peaklist_filename}' not found. Starting with an empty database.")

    lower_energies = _load_lower_state_energies(lower_states_file)
    if lower_energies is None: return # Stop if file failed to load

    # Write data summary log to Gen_Output.txt ---
    with open("Gen_Output.txt", "a") as f_log:
        f_log.write("# --- Input Data Summary ---\n")
        f_log.write(f"# Loaded {len(calculated_linelist_data)} entries from calculated linelist.\n")
        f_log.write(f"# Loaded {len(database_data)} peaks from existing database.\n")
        f_log.write(f"# Loaded {len(lower_energies)} known lower state energy levels.\n")


    # Filter calculated linelist to exclude already-assigned transitions ###
    print("--- Filtering for New Transitions ---")
    
    # Step 1: Create a set of all unique transitions already assigned in the database.
    # A set provides extremely fast lookups.
    existing_assigned_transitions = set()
    for peak in database_data:
        for assignment in peak.get('assignments', []):
            # A transition is uniquely defined by its lower and upper state QNs.
            transition_tuple = (assignment['lower_qns'], assignment['upper_qns'])
            existing_assigned_transitions.add(transition_tuple)
            
    print(f"Found {len(existing_assigned_transitions)} unique transitions already assigned in the database.")

    # Step 2: Build a new list containing only the calculated transitions that are NOT in the set.
    new_transitions_to_process = []
    for calc_entry in calculated_linelist_data:
        transition_tuple = (calc_entry['lower_qns'], calc_entry['upper_qns'])
        if transition_tuple not in existing_assigned_transitions:
            new_transitions_to_process.append(calc_entry)
            
    print(f"Filtering complete. {len(new_transitions_to_process)} new transitions will be processed.")
    
    # Log the filtering result
    with open("Gen_Output.txt", "a") as f_log:
        f_log.write("# --- Transition Filtering ---\n")
        f_log.write(f"# Found {len(existing_assigned_transitions)} existing assignments in the database.\n")
        f_log.write(f"# The analysis will proceed with {len(new_transitions_to_process)} unassigned transitions from the linelist.\n")

    # END OF FILTERING BLOCK

    # --- 2. Sort the NEW, FILTERED Linelist by Descending Intensity ---
    new_transitions_to_process.sort(key=lambda e: e['intensity'], reverse=True)

    # --- 3. Declare data_window Variable ---
    data_window = 2 * fitting_window

    print(f"Using fitting window={fitting_window}, assignment window={assignment_window}, CombDif Threshold={comb_dif_thresh}")

# --- 4. Main Loop ---
    print("--- Starting main processing loop with iterative fitting ---")
    experimental_y_working = np.copy(y)

    for i, calc_entry in enumerate(new_transitions_to_process):
        print(f"\n--- Processing Calculated Entry #{i+1}: Center={calc_entry['center']:.4f}, Intensity={calc_entry['intensity']:.2e} ---")
        
        # STEP 1: Always run the deconvolution engine for the current region.
        # This ensures the local database is as accurate as possible before assignment.
        database_data = _deconvolve_and_fit_region(
            center_freq=calc_entry['center'],
            db_data=database_data,
            exp_x=x,
            exp_y_master=experimental_y_working,
            fitting_w=fitting_window,
            data_w=data_window
        )
        
        # STEP 2: Attempt to assign the current calculated entry (calc_entry).
        assignment_window_limits = (calc_entry['center'] - assignment_window / 2, calc_entry['center'] + assignment_window / 2)
        candidate_peaks_for_T1 = [
            p for p in database_data 
            if assignment_window_limits[0] <= p['center'] <= assignment_window_limits[1] 
            and p['amplitude'] >= 0.8 * calc_entry['intensity']
        ]

        if not candidate_peaks_for_T1:
            print(f"  - Assignment SKIPPED: No suitable experimental candidate peaks found for this transition.")
            continue

        # Check if any assignments to this upper state already exist (our "anchor").
        existing_assignments = [
            p for p in database_data 
            for assign in p.get('assignments', []) 
            if assign['upper_qns'] == calc_entry['upper_qns']
        ]

        # --- CASE 1: Anchor-based assignment (Existing logic) ---
        if existing_assignments:
            upper_energies = []
            for p_assigned in existing_assignments:
                for assign in p_assigned['assignments']:
                    if assign['upper_qns'] == calc_entry['upper_qns'] and assign['lower_qns'] in lower_energies:
                        upper_e = lower_energies[assign['lower_qns']] + p_assigned['center']
                        upper_energies.append(upper_e)
            
            if upper_energies:
                avg_upper_e = np.mean(upper_energies)
                best_candidate = None
                min_deviation = float('inf')

                for cand in candidate_peaks_for_T1:
                    if calc_entry['lower_qns'] in lower_energies:
                        cand_upper_e = lower_energies[calc_entry['lower_qns']] + cand['center']
                        deviation = abs(cand_upper_e - avg_upper_e)
                        if deviation < min_deviation:
                            min_deviation = deviation
                            best_candidate = cand
                
            if best_candidate and min_deviation < assignment_window:
                # Find the peak in the main database to modify it
                for peak in database_data:
                    if np.isclose(peak['center'], best_candidate['center']):
                        new_assignment = {'lower_qns': calc_entry['lower_qns'], 'upper_qns': calc_entry['upper_qns']}
                        
                        # Check if this exact assignment already exists for this peak
                        if new_assignment not in peak.get('assignments', []):
                            if peak.get('assignments'): # Log a warning if it's becoming a blend
                                print(f"  - WARNING: Assigning to an already-assigned peak at {peak['center']:.4f}")
                            peak['assignments'].append(new_assignment)
                            print(f"  - ASSIGNED (Anchor): {calc_entry['lower_qns']} -> {calc_entry['upper_qns']} to peak at {peak['center']:.4f}")
                        else:
                            print(f"  - INFO: Assignment {calc_entry['lower_qns']} -> {calc_entry['upper_qns']} already exists for peak at {peak['center']:.4f}. Skipping.")
                        
                        break # Found and processed, exit inner loop
        
        # --- CASE 2: No anchor exists. Use Combination Difference Validation. ---
        else:
            print(f"  - No existing assignments for upper state {calc_entry['upper_qns']}. Initiating Combination Difference search...")
            
            # Step 2.1: Find a suitable partner transition (T2)
            partner_entry = _find_partner_transition(calc_entry, new_transitions_to_process, lower_energies)

            if not partner_entry:
                print("  - ComboDiff FAILED: Could not find a suitable partner transition with known lower state.")
                continue

            print(f"  - Found Partner (T2): Center={partner_entry['center']:.4f}, {partner_entry['lower_qns']} -> {partner_entry['upper_qns']}")

            # Step 2.2: Deconvolve the partner's spectral region to get accurate candidates for T2.
            database_data = _deconvolve_and_fit_region(
                center_freq=partner_entry['center'],
                db_data=database_data,
                exp_x=x,
                exp_y_master=experimental_y_working,
                fitting_w=fitting_window,
                data_w=data_window
            )
            
            partner_assignment_window = (partner_entry['center'] - assignment_window / 2, partner_entry['center'] + assignment_window / 2)
            candidate_peaks_for_T2 = [
                p for p in database_data
                if partner_assignment_window[0] <= p['center'] <= partner_assignment_window[1]
            ]
            
            if not candidate_peaks_for_T2:
                print("  - ComboDiff FAILED: No experimental candidate peaks found for partner transition T2.")
                continue

            # Step 2.3: The Combination Difference Search
            L1_qns = calc_entry['lower_qns']
            L2_qns = partner_entry['lower_qns']
            delta_E_lower_known = lower_energies[L2_qns] - lower_energies[L1_qns]
            
            best_pair_found = None
            min_of_max_positional_deviation = float('inf') 
            final_combo_diff_deviation = float('inf')

            for p1 in candidate_peaks_for_T1:
                for p2 in candidate_peaks_for_T2:
                    # Step 2.3.1: First, check if the pair satisfies the combination difference threshold.
                    # This acts as a gateway condition.
                    delta_freq_exp = p1['center'] - p2['center']
                    combo_diff_deviation = abs(delta_freq_exp - delta_E_lower_known)
                    
                    if combo_diff_deviation < comb_dif_thresh:
                        # If the pair is valid, then proceed to the NEW heuristic.
                        
                        # Step 2.3.2: Calculate the positional deviation for each peak in the pair.
                        positional_dev_p1 = abs(p1['center'] - calc_entry['center'])
                        positional_dev_p2 = abs(p2['center'] - partner_entry['center'])
                        
                        # Step 2.3.3: Find the largest of the two positional deviations.
                        largest_positional_dev_for_this_pair = max(positional_dev_p1, positional_dev_p2)
                        
                        # Step 2.3.4: Check if this pair is better than the best one found so far.
                        # "Better" is now defined as having a smaller "largest positional deviation".
                        if largest_positional_dev_for_this_pair < min_of_max_positional_deviation:
                            # This is the new best pair. Update all tracking variables.
                            min_of_max_positional_deviation = largest_positional_dev_for_this_pair
                            best_pair_found = (p1, p2)
                            # Also store the combo diff deviation of this winning pair for logging.
                            final_combo_diff_deviation = combo_diff_deviation

            # Step 2.4: Validation and Assignment (uses the new tracking variables)
            if best_pair_found:
                p1_winner, p2_winner = best_pair_found
                print(f"  - ComboDiff SUCCESS: Found a valid pair.")
                print(f"    - Winning pair's ComboDiff Deviation: {final_combo_diff_deviation:.6f} (Limit: {comb_dif_thresh})")
                print(f"    - Winning pair's Min-Max Positional Deviation: {min_of_max_positional_deviation:.6f}")

                # Assign T1 to p1_winner
                for peak in database_data:
                    if np.isclose(peak['center'], p1_winner['center']):
                        new_assignment_T1 = {'lower_qns': calc_entry['lower_qns'], 'upper_qns': calc_entry['upper_qns']}
                        
                        # --- FIX: Check if this exact assignment already exists for this peak ---
                        if new_assignment_T1 not in peak.get('assignments', []):
                            if peak.get('assignments'):
                                print(f"  - WARNING: Assigning T1 to an already-assigned peak at {peak['center']:.4f} (Blend).")
                            peak['assignments'].append(new_assignment_T1)
                            print(f"  - ASSIGNED (ComboDiff T1): {calc_entry['lower_qns']} -> {calc_entry['upper_qns']} to peak at {peak['center']:.4f}")
                        else:
                            print(f"  - INFO: Assignment T1 {calc_entry['lower_qns']} -> {calc_entry['upper_qns']} already exists for peak at {peak['center']:.4f}. Skipping.")
                        
                        break
                
                # Assign T2 to p2_winner
                for peak in database_data:
                    if np.isclose(peak['center'], p2_winner['center']):
                        new_assignment_T2 = {'lower_qns': partner_entry['lower_qns'], 'upper_qns': partner_entry['upper_qns']}
                        
                        # --- FIX: Check if this exact assignment already exists for this peak ---
                        if new_assignment_T2 not in peak.get('assignments', []):
                            if peak.get('assignments'):
                                print(f"  - WARNING: Assigning T2 to an already-assigned peak at {peak['center']:.4f} (Blend).")
                            peak['assignments'].append(new_assignment_T2)
                            print(f"  - ASSIGNED (ComboDiff T2): {partner_entry['lower_qns']} -> {partner_entry['upper_qns']} to peak at {peak['center']:.4f}")
                        else:
                            print(f"  - INFO: Assignment T2 {partner_entry['lower_qns']} -> {partner_entry['upper_qns']} already exists for peak at {peak['center']:.4f}. Skipping.")
                        
                        break
            else:
                print(f"  - ComboDiff FAILED: No pair found within tolerance. Best deviation was {min_deviation:.6f}.")

    # --- 5. Final Save and Cleanup ---
    _write_database_to_file(database_data, peaklist_filename)

    # Write finishing log message to Gen_Output.txt ---
    with open("Gen_Output.txt", "a") as f_log:
        f_log.write("# --- Simple Calculation-Assisted Rovibrational Analysis Finished ---\n\n")
    print("--- Automatic analysis finished ---")
    messagebox.showinfo("Analysis Complete", "The automatic analysis has finished processing.")

def run_new_automatic_analysis(spectrum_file, calculated_file, lower_states_file, assignment_window, fitting_window, comb_dif_thresh):
    """
    This is the new placeholder function for the automatic analysis workflow.
    It receives all the parameters from the GUI and demonstrates that they have been
    captured correctly. The actual analysis logic should be implemented here.

    Args:
        spectrum_file (str): Path to the main experimental spectrum file.
        calculated_file (str): Path to the calculated linelist file.
        lower_states_file (str): Path to the file with lower state energy levels.
        assignment_window (float): The tolerance for making assignments.
        fitting_window (float): The width of the local region for deconvolution and fitting.
        comb_dif_thresh (float): The specific tolerance for validating a combination difference.
    """
    # Log the received parameters to the console for verification
    print("--- Starting New Automatic Analysis Workflow ---")
    print(f"  > Main Spectrum File: {spectrum_file}")
    print(f"  > Calculated Linelist File: {calculated_file}")
    print(f"  > Lower States File: {lower_states_file}")
    print("-" * 20)
    print(f"  > Assignment Window: {assignment_window}")
    print(f"  > Fitting Window: {fitting_window}")
    print(f"  > Comb. Difference Threshold: {comb_dif_thresh}")
    print("-------------------------------------------------")

    # Provide feedback to the user via a message box
    messagebox.showinfo(
        "Workflow Started",
        "The new automatic analysis workflow has started.\n\n"
        "Parameters have been printed to the console for verification.\n"
        "Implement your new logic in the 'run_new_automatic_analysis' function."
    )

def basic_automatic_rovibrational(calc_linelist_file, lower_states_filepath, assignment_window, fitting_window, comb_dif_thresh):
    """
    Performs a basic automatic rovibrational analysis.

    This function reads a calculated linelist and a file of lower state energy levels,
    preparing the data for subsequent analysis steps.

    Args:
        calc_linelist_file (str): Path to the calculated linelist file.
        lower_states_filepath (str): Path to the file with lower state energy levels.
        assignment_window (float): The tolerance for making assignments.
        fitting_window (float): The width of the local region for deconvolution.
        comb_dif_thresh (float): The tolerance for validating combination differences.
    """
    print("--- Starting Basic Automatic Rovibrational Analysis ---")
    print(f"  - Assignment Window: {assignment_window}")
    print(f"  - Fitting Window: {fitting_window}")
    print(f"  - Combination Difference Threshold: {comb_dif_thresh}")
    
    # --- Step 1: Read Calculated Linelist File ---
    calculated_transitions = []
    print(f"\n[Step 1] Reading calculated linelist from: {calc_linelist_file}")
    
    try:
        with open(calc_linelist_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Skip empty or commented lines
                if not line.strip() or line.strip().startswith('#'):
                    continue
                
                parts = line.strip().split()
                
                # Validate that the line has exactly 10 columns
                if len(parts) != 10:
                    print(f"  - WARNING: Skipping line {line_num}. Expected 10 columns, but found {len(parts)}.")
                    continue
                
                try:
                    # Parse all 10 columns
                    entry = {
                        'center': float(parts[0]),
                        'intensity': float(parts[1]),
                        'lower_qns': (int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])), # J", Ka", Kc", v"
                        'upper_qns': (int(parts[6]), int(parts[7]), int(parts[8]), int(parts[9]))  # J', Ka', Kc', v'
                    }
                    calculated_transitions.append(entry)
                
                except (ValueError, IndexError) as e:
                    print(f"  - WARNING: Skipping line {line_num} due to data conversion error: {e}. Line was: '{line.strip()}'")
                    continue
        
        print(f"  - Successfully loaded {len(calculated_transitions)} transitions from the calculated linelist.")

    except FileNotFoundError:
        messagebox.showerror("File Not Found", f"The calculated linelist file could not be found:\n{calc_linelist_file}")
        print(f"  - ERROR: File not found at '{calc_linelist_file}'. Aborting analysis.")
        return # Stop the function if the file doesn't exist

    except Exception as e:
        messagebox.showerror("Error Reading File", f"An unexpected error occurred while reading the calculated linelist:\n{e}")
        print(f"  - ERROR: An unexpected error occurred: {e}. Aborting analysis.")
        return

    # --- Step 2: Read Lower State Energy Levels File ---
    lower_state_energies = {}
    print(f"\n[Step 2] Reading lower state energies from: {lower_states_filepath}")

    try:
        with open(lower_states_filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Skip empty or commented lines
                if not line.strip() or line.strip().startswith('#'):
                    continue
                
                parts = line.strip().split()

                # Validate that the line has exactly 5 columns
                if len(parts) != 5:
                    print(f"  - WARNING: Skipping line {line_num}. Expected 5 columns, but found {len(parts)}.")
                    continue
                
                try:
                    # The key for the dictionary is the tuple of quantum numbers
                    qns_key = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])) # J", Ka", Kc", v"
                    
                    # The value is the energy
                    energy_value = float(parts[0])

                    # Check for duplicate quantum number states
                    if qns_key in lower_state_energies:
                        print(f"  - WARNING: Duplicate lower state QNs found on line {line_num}: {qns_key}. The previous entry will be overwritten.")
                    
                    lower_state_energies[qns_key] = energy_value

                except (ValueError, IndexError) as e:
                    print(f"  - WARNING: Skipping line {line_num} due to data conversion error: {e}. Line was: '{line.strip()}'")
                    continue
        
        print(f"  - Successfully loaded {len(lower_state_energies)} unique lower state energy levels.")

    except FileNotFoundError:
        messagebox.showerror("File Not Found", f"The lower states file could not be found:\n{lower_states_filepath}")
        print(f"  - ERROR: File not found at '{lower_states_filepath}'. Aborting analysis.")
        return # Stop the function if the file doesn't exist

    except Exception as e:
        messagebox.showerror("Error Reading File", f"An unexpected error occurred while reading the lower states file:\n{e}")
        print(f"  - ERROR: An unexpected error occurred: {e}. Aborting analysis.")
        return

    #
    # >>> The next steps of your analysis logic will go here. <<<
    # You now have `calculated_transitions` (a list of dictionaries) and
    # `lower_state_energies` (a dictionary) available for use.
    #
    print("\n--- Data loading and parsing complete. ---")

def automatic_rovibrational_analysis():
    """
    Opens the dialog for the 'Automatic rotational vibrational analysis' module.
    This window allows the user to specify files and parameters for the analysis.
    """
    global auto_analysis_window_open, filename # filename is the main spectrum file

    # Prevent opening multiple instances of this window
    if auto_analysis_window_open:
        return
    auto_analysis_window_open = True

    # --- Window and Frame Setup ---
    # Find the parent window for the dialog
    fig_manager = plt.get_current_fig_manager()
    root = fig_manager.window if fig_manager and hasattr(fig_manager, 'window') else tk.Tk()

    dialog = Toplevel(root)
    dialog.title("Automatic Rovibrational Analysis")
    position_dialog_near_cursor(dialog, width=700, height=400)

    # --- Create Layout Frames ---
    content_container = Frame(dialog)
    content_container.pack(fill=BOTH, expand=True, padx=10, pady=5)

    bottom_button_frame = Frame(dialog)
    bottom_button_frame.pack(side=tk.BOTTOM, fill='x', padx=10, pady=10)

    # Frame for the UI elements, with a title
    workflow_frame = Frame(content_container, bd=2, relief="groove")
    workflow_frame.pack(fill=BOTH, expand=True)
    
    Label(workflow_frame, text="Simple Calculation-Assisted Workflow", font=("Arial", 12, "bold")).pack(pady=(10, 15))

    # --- UI Elements for File Selection ---
    
    # Use a StringVar to hold the path of the selected file
    dialog.calc_linelist_path = tk.StringVar(master=dialog)

    def _select_calc_linelist_file():
        """Opens a file dialog and updates the path variable."""
        filepath = filedialog.askopenfilename(
            title="Select Calculated Linelist File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            parent=dialog
        )
        if filepath:
            dialog.calc_linelist_path.set(filepath)

    def _show_linelist_help():
        """Displays a help message box for the linelist file format."""
        help_text = (
            "Calculated linelist file structure in columns (delimiter - space):\n\n"
            "line center, peak maximum, contour type (Gaussian, Lorentzian and Voigt), "
            "FWHM (one for gaussian and lorentzian and both for voigt), "
            "lower quantum numbers, upper quantum numbers"
        )
        messagebox.showinfo("Calculated Linelist Format", help_text, parent=dialog)

    # Create a sub-frame to hold the file selection widgets on one line
    file_selection_frame = Frame(workflow_frame)
    file_selection_frame.pack(fill='x', padx=10, pady=10)

    Button(
        file_selection_frame, 
        text="Calculated linelist", 
        command=_select_calc_linelist_file
    ).pack(side=LEFT, padx=(0, 5))

    Label(
        file_selection_frame, 
        textvariable=dialog.calc_linelist_path, 
        relief="sunken", 
        anchor='w',
        justify=LEFT
    ).pack(side=LEFT, fill='x', expand=True)

    Button(
        file_selection_frame,
        text="?",
        command=_show_linelist_help,
        font=("Arial", 8, "bold"),
        padx=1
    ).pack(side=LEFT, padx=(5, 0))
    
    # Frame for the Lower States file selection
    lower_states_frame = Frame(workflow_frame)
    lower_states_frame.pack(fill='x', padx=10, pady=(0, 10))

    dialog.lower_states_path = tk.StringVar(master=dialog)

    def _select_lower_states_file():
        """Opens a file dialog for the lower state energy levels file."""
        filepath = filedialog.askopenfilename(
            title="Select Lower State Energy Levels File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            parent=dialog
        )
        if filepath:
            dialog.lower_states_path.set(filepath)

    def _show_lower_states_help():
        """Displays a help message for the lower states file format."""
        help_text = (
            "Lower state levels file with the following format:\n\n"
            "energy, quantum numbers\n\n"
            "All data should be separated with spaces."
        )
        messagebox.showinfo("Lower States File Format", help_text, parent=dialog)

    Button(
        lower_states_frame,
        text="Lower states",
        command=_select_lower_states_file
    ).pack(side=LEFT, padx=(0, 5))

    Label(
        lower_states_frame,
        textvariable=dialog.lower_states_path,
        relief="sunken",
        anchor='w',
        justify=LEFT
    ).pack(side=LEFT, fill='x', expand=True)

    Button(
        lower_states_frame,
        text="?",
        command=_show_lower_states_help,
        font=("Arial", 8, "bold"),
        padx=1
    ).pack(side=LEFT, padx=(5, 0))

    # --- UI Elements for Algorithm Parameters ---
    params_frame = Frame(workflow_frame)
    params_frame.pack(fill='x', padx=10, pady=10)
    params_frame.columnconfigure(1, weight=1)
    params_frame.columnconfigure(3, weight=1)

    Label(params_frame, text="Algorithm Parameters", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 5))

    dialog.assignment_window_var = tk.StringVar(master=dialog, value="0.003")
    dialog.fitting_window_var = tk.StringVar(master=dialog, value="0.01")
    dialog.comb_dif_thresh_var = tk.StringVar(master=dialog, value="0.003")

    Label(params_frame, text="Assignment Window:").grid(row=1, column=0, sticky="w", padx=5)
    assignment_entry = Entry(params_frame, width=15, textvariable=dialog.assignment_window_var)
    assignment_entry.grid(row=1, column=1, sticky="ew")

    Label(params_frame, text="Fitting Window:").grid(row=1, column=2, sticky="w", padx=(10, 5))
    fitting_entry = Entry(params_frame, width=15, textvariable=dialog.fitting_window_var)
    fitting_entry.grid(row=1, column=3, sticky="ew")

    Label(params_frame, text="Threshold CombDif:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
    comb_dif_entry = Entry(params_frame, width=15, textvariable=dialog.comb_dif_thresh_var)
    comb_dif_entry.grid(row=2, column=1, sticky="ew", pady=5)

    # --- "Run" Button with Placeholder Logic ---
    run_button_frame = Frame(workflow_frame)
    run_button_frame.pack(side=BOTTOM, fill='x', padx=10, pady=(10, 5))

    def _run_new_analysis_placeholder():
        """
        This is a placeholder function for your new analysis logic.
        It gathers and validates all the inputs from the GUI.
        """
        print("\n--- Triggering New Automatic Analysis ---")
        
        # 1. Gather file paths
        main_spectrum_file = filename
        calc_linelist_file = dialog.calc_linelist_path.get()
        lower_states_filepath = dialog.lower_states_path.get()
        
        # 2. Validate file paths
        if not main_spectrum_file or not calc_linelist_file or not lower_states_filepath:
            messagebox.showerror("Missing File", "Please provide all required files: main spectrum, calculated linelist, and lower states.", parent=dialog)
            return
            
        # 3. Gather and validate numerical parameters
        try:
            assignment_val = float(dialog.assignment_window_var.get())
            fitting_val = float(dialog.fitting_window_var.get())
            comb_dif_val = float(dialog.comb_dif_thresh_var.get())

            if not (assignment_val > 0 and fitting_val > 0 and comb_dif_val > 0):
                raise ValueError("Window and threshold values must be positive.")

        except ValueError as e:
            messagebox.showerror("Invalid Parameter", f"Please enter valid positive numbers for the algorithm parameters.\n\nError: {e}", parent=dialog)
            return

        # 4. Print the gathered parameters to the console
        print(f"  - Main Spectrum File: {main_spectrum_file}")
        print(f"  - Calculated Linelist File: {calc_linelist_file}")
        print(f"  - Lower States File: {lower_states_filepath}")
        print(f"  - Assignment Window: {assignment_val}")
        print(f"  - Fitting Window: {fitting_val}")
        print(f"  - Combination Difference Threshold: {comb_dif_val}")
        print("----------------------------------------")
        
        # >>> YOUR NEW ANALYSIS LOGIC WOULD GO HERE <<<
        # Call the new function with the validated parameters.
        basic_automatic_rovibrational(
            calc_linelist_file,
            lower_states_filepath,
            assignment_val,
            fitting_val,
            comb_dif_val
        )

    Button(
        run_button_frame, 
        text="Run", 
        command=_run_new_analysis_placeholder, 
        font=("Arial", 10, "bold")
    ).pack(side=RIGHT)

    # --- Bottom Frame: Dialog Buttons ---
    def close_dialog_and_flag():
        """Handles closing the dialog and resetting the global flag."""
        nonlocal dialog
        globals()['auto_analysis_window_open'] = False
        dialog.destroy()

    Button(bottom_button_frame, text="Close", command=close_dialog_and_flag).pack(side=RIGHT, padx=5)

    # --- Finalize and Show Window ---
    dialog.protocol("WM_DELETE_WINDOW", close_dialog_and_flag)

    dialog.lift()
    dialog.attributes('-topmost', True)
    dialog.after_idle(dialog.attributes, '-topmost', False)
    dialog.focus_force()

if __name__ == "__main__":
    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    fig.canvas.mpl_connect('pick_event', on_calc_pick)

    # The code to add the Tkinter "Settings" button
    try:
        fig_manager = plt.get_current_fig_manager()
        if fig_manager is not None and hasattr(fig_manager, 'window'):
            root_window = fig_manager.window
            settings_button = tk.Button(master=root_window, text="Settings", command=show_settings_dialog)
            settings_button.place(x=10, y=10)
    except Exception as e:
        print(f"Warning: Failed to create the 'Settings' button. Error: {e}")

    # Force initial axis synchronization before displaying the plot ---
    # This corrects any desynchronization that happens during the initial plot setup.
    initial_xlim = ax1.get_xlim()
    ax2.set_xlim(initial_xlim)
    ax3.set_xlim(initial_xlim) # Also sync the residual plot for good measure
    initial_ylim = ax1.get_ylim()
    ax2.set_ylim(initial_ylim)
    ax3.set_ylim(initial_ylim) # Also sync the residual plot for good measure

    show_settings_dialog()
    plt.show()