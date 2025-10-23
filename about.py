# about.py
# A comprehensive user guide and operational manual for COROVA.
# This dictionary contains the text for the multi-page "About program" manual.

aboutprogram = {
            "General overview": (
                """COROVA: A COde for ROtational-Vibrational Analysis

Welcome to COROVA, an integrated software environment for the analysis of high-resolution spectroscopic data. This program is designed to bridge the gap between raw experimental spectra and a fully characterized, quantum-mechanically assigned linelist.

--- PROGRAM PHILOSOPHY ---
COROVA is built upon a workflow-centric design. The core philosophy is to combine robust algorithms with intuitive, interactive controls, allowing for both rapid analysis of large datasets and detailed manual interrogation of individual spectral features.
                """
            ),
            "Controls": (
                """INTERACTIVE PLOT CONTROLS

Efficient operation within COROVA is facilitated by a set of mouse and keyboard controls for direct data manipulation within the main plot window.

--- MOUSE CONTROLS ---
  - Left-Click and Drag: Pans the spectral view along both axes.
  - Right-Click on Peak (*): Selects the nearest peak marker and opens the 'Edit Peak & Assign QNs' dialog for detailed modification and assignment.
  - Mouse Wheel: Zooms the view, centered on the cursor's current position.

--- Primary KEYBOARD Shortcuts ---
  'T' - TAG PEAK: Adds a new peak to the 'Peaklist.txt' file at the current cursor coordinates. This serves as an initial guess for the fitting engine.

  'C' - DELETE PEAK: Removes the peak visually closest to the cursor from the database file.

  'E' - SET FITTING LIMITS: Defines a fitting region with two vertical markers. A third press clears the region.

  'Z' - PERFORM CONTOUR FITTING: Initiates the Peak Fitting Engine for all tagged peaks within the defined region.

--- NAVIGATION Shortcuts ---
  - Arrow Keys: Incrementally pan the view.
  - 'A': Move the view to the left.
  - 'D': Move the view to the right.
  - '+' and '-': Zoom the X-axis (frequency).
  - '[' and ']': Zoom the Y-axis (intensity). This is context-sensitive: if the cursor is over the residual plot, it only zooms that plot's Y-axis.

--- IMPLICIT Matplotlib Controls ---
  The plot window also responds to standard Matplotlib shortcuts:
  - 'S': Opens a 'Save File' dialog.
  - 'F': Toggles fullscreen mode.
  - 'G': Toggles the plot grid on/off.
  - 'H' or 'Home': Resets the view to its original state.
  - 'Q': Quits the program.
                """
            ),
            "Credits": (
                """--- CREDITS & ACKNOWLEDGEMENTS ---

COROVA was designed and developed by Egor O. Dobrolyubov, 2025.

THIRD-PARTY LIBRARIES:
  COROVA is built on the foundation of the extensive Python scientific ecosystem. We gratefully acknowledge the developers of the following essential libraries:
  - NumPy: For numerical operations.
  - SciPy: For scientific algorithms (filtering, optimization, etc.).
  - Matplotlib: For all data visualization and plotting.
  - Tkinter: For the graphical user interface.
  - Joblib: For parallel processing in the fitting engine.
                """
            )
        }