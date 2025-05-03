
 /**
  * 3D UMAP Visualization with Interactive Selection
  * This script creates an interactive 3D visualization of UMAP data
  * with point selection capabilities.
  */

 // ======== Configuration Constants ========
 const CONFIG = {
    // Visualization settings
    POINT_SIZE: 0.05,
    SCALE_FACTOR: 5,

    // Camera settings
    CAMERA: {
       FOV: 75,
       NEAR: 0.1,
       FAR: 1000,
       POSITION: [0, 0, 10]
    },

    // Controls settings
    CONTROLS: {
       DAMPING_FACTOR: 0.03,
       MIN_DISTANCE: 1,
       MAX_DISTANCE: 20
    },

    // Animation settings
    ANIMATION: {
       DURATION: 7,
       ORBIT_RADIUS: 10
    },

    // Data settings
    DATA: {
       DEFAULT_FILES: ["umap_data.json", "umap_data_100k.json"],
       DEFAULT_FILE_INDEX: 0
    },

    // UI messages
    MESSAGES: {
       LOADING: "Loading data...",
       ROTATING: "Rotating camera for overview...",
       SELECTION_ON: "Selection Mode ON - Click and drag to select points",
       SELECTION_OFF: "Selection Mode OFF",
       NO_POINTS: "No points found in the loaded data.",
       NO_FILES: "No JSON files found!"
    }
 };

 // ======== State Variables ========
 const state = {
    // Three.js objects
    scene: null,
    camera: null,
    renderer: null,
    pointCloud: null,
    controls: null,

    // Selection state
    isSelecting: false,
    selectionModeActive: false,
    isSKeyPressed: false,
    selectionStart: { x: 0, y: 0 },
    selectionEnd: { x: 0, y: 0 },
    selectedPoints: [],

    // Data state
    points: [],
    originalColors: [],
    currentColors: [],

    // Currently visible columns
    visibleColumns: [] 
 };

 // ======== DOM Elements Cache ========
 const elements = {
    selectionBox: document.getElementById("selectionBox"),
    infoBox: document.getElementById("info"),
    messageBox: document.getElementById("rotation-message"),
    errorBox: document.getElementById("error-message"),
    selectionStatus: document.getElementById("selection-status"),
    resetSelection: document.getElementById("reset-selection"),
    jsonFileSelect: document.getElementById("jsonFileSelect"),
    colorColumnSelect: document.getElementById("colorColumnSelect"),
    minimizeButton: document.getElementById("minimize-button"),
    controlsBox: document.getElementById("controls-box"),
    controlsTab: document.getElementById("controls-tab")
 };

 // ======== Initialization ========
 /**
  * Initialize the application
  */
 function init() {
    console.log("Initializing application...");

    // Set up Three.js scene
    initScene();

    // Set up event listeners
    initEventListeners();

    // Set up minimize controls
    initMinimizeControls();

    // Fetch the list of JSON files and load the first available one
    fetchJSONList();

    // Start rendering loop
    animate();

    // Trigger initial camera rotation animation
    rotateCameraAnimation(CONFIG.ANIMATION.DURATION);

    console.log("Initialization complete");
 }

 /**
  * Initialize the Three.js scene, camera, renderer and controls
  */
 function initScene() {
    // Create a new scene
    state.scene = new THREE.Scene();

    // Set up camera
    state.camera = new THREE.PerspectiveCamera(
       CONFIG.CAMERA.FOV, 
       window.innerWidth / window.innerHeight, 
       CONFIG.CAMERA.NEAR, 
       CONFIG.CAMERA.FAR
    );
    state.camera.position.set(...CONFIG.CAMERA.POSITION);

    // Set up renderer
    state.renderer = new THREE.WebGLRenderer();
    state.renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(state.renderer.domElement);

    // Add orbit controls for camera movement
    state.controls = new THREE.OrbitControls(state.camera, state.renderer.domElement);
    state.controls.enableDamping = true;
    state.controls.dampingFactor = CONFIG.CONTROLS.DAMPING_FACTOR;
    state.controls.minDistance = CONFIG.CONTROLS.MIN_DISTANCE;
    state.controls.maxDistance = CONFIG.CONTROLS.MAX_DISTANCE;
 }

 /**
  * Initialize all event listeners
  */
 function initEventListeners() {   

    // Reset Selection Button Press
    elements.resetSelection.addEventListener("click", resetSelection);

    // Mouse events - use capture phase (true) to ensure handlers run before Three.js controls
    window.addEventListener("mousedown", onMouseDown, true);
    window.addEventListener("mousemove", onMouseMove, true);
    window.addEventListener("mouseup", onMouseUp, true);

    // Keyboard events
    window.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("keyup", onKeyUp, true);


    // Window resize
    window.addEventListener("resize", onWindowResize, false);

    // Column Settings toggle
    document.getElementById('column-settings-toggle').addEventListener('click', toggleColumnSettingsPanel);

    console.log("Event listeners initialized");
 }


 function initMinimizeControls() {
   // Check if there's a saved preference
   const isPanelMinimized = localStorage.getItem('controlsPanelMinimized') === 'true';

   // Apply initial state
   if (isPanelMinimized) {
      elements.controlsBox.classList.add('minimized');
      elements.controlsTab.classList.remove('hidden');
   }else {
  // Explicitly ensure controls tab is hidden in normal state
  elements.controlsBox.classList.remove('minimized');
  elements.controlsTab.classList.add('hidden');
   }

   // Set up minimize button click handler
   elements.minimizeButton.addEventListener('click', minimizePanel);

   // Set up tab click handler to expand the panel
   elements.controlsTab.addEventListener('click', maximizePanel);

   console.log("Minimize controls initialized");
}

/**
 * Minimize the controls panel
 */
function minimizePanel() {
   elements.controlsBox.classList.add('minimized');
   elements.controlsTab.classList.remove('hidden');
   localStorage.setItem('controlsPanelMinimized', 'true');
}

/**
 * Maximize the controls panel
 */
function maximizePanel() {
   elements.controlsBox.classList.remove('minimized');
   elements.controlsTab.classList.add('hidden');
   localStorage.setItem('controlsPanelMinimized', 'false');
} 





 // ======== Event Handlers ========
 /**
  * Handle key down events
  * @param {KeyboardEvent} event - The keyboard event
  */
 function onKeyDown(event) {
    if (event.key.toLowerCase() === "s") {
       state.isSKeyPressed = true;
       state.selectionModeActive = true;

       // Update UI to reflect selection mode is active
       elements.selectionStatus.textContent = "Selection Mode: ON (via S key)";
       elements.selectionStatus.classList.add("active");

       // Show a brief message to the user
       showMessage(CONFIG.MESSAGES.SELECTION_ON);
       setTimeout(hideMessage, 2000);

       console.log("S key pressed - Selection mode active");

       // Prevent browser default behavior
       event.preventDefault();
       event.stopPropagation();

       // Make sure controls are still enabled when not actively selecting
       if (!state.isSelecting) {
          state.controls.enabled = true;
       }
    }
 }

 /**
  * Handle key up events
  * @param {KeyboardEvent} event - The keyboard event
  */
 function onKeyUp(event) {
    if (event.key.toLowerCase() === "s") {
       state.isSKeyPressed = false;

       // Always deactivate selection mode when S key is released
       // if it was activated by the S key
       state.selectionModeActive = false;
       elements.selectionStatus.textContent = "Selection Mode: OFF";
       elements.selectionStatus.classList.remove("active");
       showMessage(CONFIG.MESSAGES.SELECTION_OFF);
       setTimeout(hideMessage, 1000);

       // Re-enable orbit controls
       state.controls.enabled = true;

       // If we were in the middle of a selection, cancel it
       if (state.isSelecting) {
          state.isSelecting = false;
          elements.selectionBox.style.display = "none";
       }

       console.log("S key released - Selection mode deactivated");

       // Prevent browser default behavior
       event.preventDefault();
       event.stopPropagation();
    }
 }

 /**
  * Handle mouse down event
  * @param {MouseEvent} event - The mouse event
  */
 function onMouseDown(event) {
    // Only process left mouse button
    if (event.button !== 0) return;

    // Check if selection mode is active (either via S key or toggle button)
    if (state.selectionModeActive || state.isSKeyPressed) {
       console.log("Selection started");
       state.isSelecting = true;

       // IMPORTANT: Disable controls immediately to prevent camera movement
       state.controls.enabled = false;

       state.selectionStart.x = event.clientX;
       state.selectionStart.y = event.clientY;

       // Set up selection box
       elements.selectionBox.style.left = `${state.selectionStart.x}px`;
       elements.selectionBox.style.top = `${state.selectionStart.y}px`;
       elements.selectionBox.style.width = "0px";
       elements.selectionBox.style.height = "0px";
       elements.selectionBox.style.display = "block";

       // Stop event propagation
       event.preventDefault();
       event.stopPropagation();
    }
 }

 /**
  * Handle mouse move event
  * @param {MouseEvent} event - The mouse event
  */
 function onMouseMove(event) {
    // Only update the selection box if we're in the middle of selecting
    if (!state.isSelecting) return;

    // Update end coordinates
    state.selectionEnd.x = event.clientX;
    state.selectionEnd.y = event.clientY;

    // Ensure selection box is visible and properly sized
    elements.selectionBox.style.display = "block";
    elements.selectionBox.style.left = `${Math.min(state.selectionStart.x, state.selectionEnd.x)}px`;
    elements.selectionBox.style.top = `${Math.min(state.selectionStart.y, state.selectionEnd.y)}px`;
    elements.selectionBox.style.width = `${Math.abs(state.selectionEnd.x - state.selectionStart.x)}px`;
    elements.selectionBox.style.height = `${Math.abs(state.selectionEnd.y - state.selectionStart.y)}px`;

    console.log(`Drawing selection box: ${Math.abs(state.selectionEnd.x - state.selectionStart.x)} × ${Math.abs(state.selectionEnd.y - state.selectionStart.y)}`);

    // Prevent default behavior to avoid camera movement
    event.preventDefault();
    event.stopPropagation();
 }

 /**
  * Handle mouse up event
  * @param {MouseEvent} event - The mouse event
  */
 function onMouseUp(event) {
    // If we're not currently selecting, exit early
    if (!state.isSelecting) return;

    console.log("Selection ended");

    // End selection mode
    state.isSelecting = false;
    elements.selectionBox.style.display = "none";

    // Re-enable orbit controls
    state.controls.enabled = true;

    // Check if we have valid start and end points
    if (typeof state.selectionStart.x === 'undefined' || typeof state.selectionEnd.x === 'undefined') {
       console.warn("Invalid selection coordinates");
       return;
    }

    // Process the selection
    processSelection();

    // Prevent default behavior
    event.preventDefault();
    event.stopPropagation();
 }

 /**
 * Process the current selection box and find points within it
 */
function processSelection() {
   // Calculate normalized selection box coordinates
   const minX = (Math.min(state.selectionStart.x, state.selectionEnd.x) / window.innerWidth) * 2 - 1;
   const maxX = (Math.max(state.selectionStart.x, state.selectionEnd.x) / window.innerWidth) * 2 - 1;
   const minY = -(Math.max(state.selectionStart.y, state.selectionEnd.y) / window.innerHeight) * 2 + 1;
   const maxY = -(Math.min(state.selectionStart.y, state.selectionEnd.y) / window.innerHeight) * 2 + 1;

   // Reset selected points
   state.selectedPoints = [];

   if (!state.pointCloud) {
      console.warn("No point cloud available");
      return;
   }

   const geometry = state.pointCloud.geometry;
   const positions = geometry.attributes.position.array;
   const colors = geometry.attributes.color.array;

   // Reset all points to current coloring scheme
   for (let i = 0; i < colors.length / 3; i++) {
      if (i < state.currentColors.length) {
         colors[i * 3] = state.currentColors[i][0];
         colors[i * 3 + 1] = state.currentColors[i][1];
         colors[i * 3 + 2] = state.currentColors[i][2];
      }
   }

   // Find points in the selection box
   for (let i = 0; i < positions.length / 3; i++) {
      // Project 3D point to screen space
      const vector = new THREE.Vector3(
         positions[i * 3],
         positions[i * 3 + 1],
         positions[i * 3 + 2]
      ).project(state.camera);

      // Check if point is in selection box
      if (vector.x >= minX && vector.x <= maxX && vector.y >= minY && vector.y <= maxY) {
         // Add to selected points
         if (state.points[i] && state.points[i].id !== undefined) {
            state.selectedPoints.push(state.points[i].id);
         }

         // Change color to white (selected)
         colors[i * 3] = 1;     // Red
         colors[i * 3 + 1] = 1; // Green
         colors[i * 3 + 2] = 1; // Blue
      }
   }

   // Update color buffer
   geometry.attributes.color.needsUpdate = true;

   // Update info display
   updateInfoBox();

   // Show feedback if points were selected
   if (state.selectedPoints.length > 0) {
      showMessage(`Selected ${state.selectedPoints.length} points`);
      setTimeout(hideMessage, 2000);
   }

   console.log(`Selected ${state.selectedPoints.length} points`);
}

 /**
  * Handle window resize event
  */
 function onWindowResize() {
    state.camera.aspect = window.innerWidth / window.innerHeight;
    state.camera.updateProjectionMatrix();
    state.renderer.setSize(window.innerWidth, window.innerHeight);
 }

 // ======== Visualization Functions ========
 /**
  * Main animation loop
  */
 function animate() {
    requestAnimationFrame(animate);
    if (state.controls) state.controls.update();
    if (state.renderer && state.scene && state.camera) {
       state.renderer.render(state.scene, state.camera);
    }
 }

 /**
  * Create a color based on a value within a range
  * @param {number} value - The value to map to a color
  * @param {number} min - Minimum value in the range
  * @param {number} max - Maximum value in the range
  * @returns {Array} RGB color values as an array [r, g, b]
  */
 function calculateColormap(value, min, max) {
    // Normalize between 0 and 1
    const t = Math.max(0, Math.min(1, (value - min) / (max - min)));

    // Increased brightness for the entire Viridis spectrum
    const r = Math.max(0.4, Math.min(88 + 180 * t, 255)) / 255;
    const g = Math.max(0.5, Math.min(50 + 160 * t, 255)) / 255;
    const b = Math.max(0.6, Math.min(120 - 40 * t, 255)) / 255;

    return [r, g, b];
 }

 /**
  * Create the 3D point cloud visualization
  */
 function createPointCloud() {
    const geometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];

    // Calculate data bounds for normalization
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    state.points.forEach(p => {
       minX = Math.min(minX, p.x);
       minY = Math.min(minY, p.y);
       minZ = Math.min(minZ, p.z);
       maxX = Math.max(maxX, p.x);
       maxY = Math.max(maxY, p.y);
       maxZ = Math.max(maxZ, p.z);
    });

    const maxRange = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
    const scaleFactor = CONFIG.SCALE_FACTOR / maxRange;

    // Reset color arrays
    state.originalColors = [];
    state.currentColors = [];

    // Normalize and store point positions and colors
    state.points.forEach(point => {
       // Center and scale points
       let x = (point.x - (minX + maxX) / 2) * scaleFactor;
       let y = (point.y - (minY + maxY) / 2) * scaleFactor;
       let z = (point.z - (minZ + maxZ) / 2) * scaleFactor;

       positions.push(x, y, z);

       // Calculate colors
       const [r, g, b] = calculateColormap(point.x, minX, maxX);
       colors.push(r, g, b);

       // Store original colors
       state.originalColors.push([r, g, b]);
       state.currentColors.push([r, g, b]);
    });

    // Create geometry attributes
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

    // Create point cloud material and add to scene
    const material = new THREE.PointsMaterial({ 
       size: CONFIG.POINT_SIZE, 
       vertexColors: true,
       sizeAttenuation: true,
       alphaTest: 0.5,
       transparent: true,
       map: createCircleTexture()
    });
    state.pointCloud = new THREE.Points(geometry, material);
    state.scene.add(state.pointCloud);

    //Colorbar
    updateColorbar("x", minX, maxX);
 }

 /**
  * Camera rotation animation for better data overview
  * @param {number} duration - Animation duration in seconds
  */
 function rotateCameraAnimation(duration) {
    let startTime = performance.now();
    let radius = CONFIG.ANIMATION.ORBIT_RADIUS;

    // Show message during rotation
    showMessage(CONFIG.MESSAGES.ROTATING);

    // Disable user controls during animation
    state.controls.enabled = false;

    // Easing function for smooth animation
    function easeOutQuad(t) {
       return t * (2 - t);
    }

    // Animation loop
    function rotateLoop(time) {
       let elapsed = (time - startTime) / 1000; // Convert to seconds
       let progress = elapsed / duration;

       if (progress >= 1) {
          state.controls.enabled = true; // Re-enable controls after animation
          hideMessage();
          return;
       }

       let easedProgress = easeOutQuad(progress);
       let angle = easedProgress * Math.PI * 2; // Full rotation

       state.camera.position.x = radius * Math.cos(angle);
       state.camera.position.z = radius * Math.sin(angle);
       state.camera.lookAt(state.scene.position); // Keep camera focused on center

       requestAnimationFrame(rotateLoop);
    }

    requestAnimationFrame(rotateLoop);
 }

 // ======== UI Interaction Functions ========

 /**
 * Update info box with selected points data
 */
function updateInfoBox() {
   const footerElement = document.getElementById('selection-info-footer');
   const tableContainer = document.getElementById('selection-table-container');
   const tableHeader = document.getElementById('selection-table-header');
   const tableBody = document.getElementById('selection-table-body');
   const infoElement = document.getElementById('info');
   
   // Clear previous content
   tableHeader.innerHTML = '';
   tableBody.innerHTML = '';
   
   if (state.selectedPoints.length > 0) {
       // Add 'has-data' class to info element when points are selected
       infoElement.classList.add('has-data');

       // Update footer text with selection count
       footerElement.textContent = `Selected ${state.selectedPoints.length} points`;
       
       // Get selected point objects from the state.points array
       const selectedPointObjects = state.points.filter(point => 
           state.selectedPoints.includes(point.id)
       );
       
       if (selectedPointObjects.length > 0) {
           // Filter to only use visible columns
           const visibleColumns = state.visibleColumns.filter(col => 
               Object.keys(selectedPointObjects[0]).includes(col)
           );
           
           // Create table header with only visible columns
           visibleColumns.forEach(columnName => {
               const th = document.createElement('th');
               th.textContent = columnName;
               tableHeader.appendChild(th);
           });
           
           // Create table rows for each selected point, showing only visible columns
           selectedPointObjects.forEach(point => {
               const tr = document.createElement('tr');
               
               visibleColumns.forEach(columnName => {
                   const td = document.createElement('td');
                   
                   // Format the cell value based on its type
                   const value = point[columnName];
                   if (typeof value === 'number') {
                     if (Number.isInteger(value)) {
                        td.textContent = value;
                     } else {
                        td.textContent = value.toFixed(2);
                     }
                   } else {
                       td.textContent = value;
                   }
                   
                   tr.appendChild(td);
               });
               
               tableBody.appendChild(tr);
           });
           
           // Show the table
           tableContainer.style.display = 'block';
       }
   } else {
       // Remove 'has-data' class when no points are selected
       infoElement.classList.remove('has-data');

       // If no points selected, hide table and show default message
       footerElement.textContent = "Drag to select multiple points";
       tableContainer.style.display = 'none';

       // Hide Settings Panel
       document.getElementById('column-settings-panel').classList.add('hidden');
   }
}

  /**
  * Update the color bar on relevant events
  */
 function updateColorbar(column, minVal, maxVal) {
   // Update colorbar header
   const colorbarHeader = document.querySelector('.colorbar-header');
   colorbarHeader.textContent = `${column}`;
   
   // Update min/max labels
   document.getElementById('min-value').textContent = minVal.toFixed(2);
   document.getElementById('max-value').textContent = maxVal.toFixed(2);
   
   // Update gradient - recreate the same gradient used in calculateColormap
   const gradientStops = [];
   for (let i = 0; i <= 10; i++) {
     const t = i / 10;
     const r = Math.max(0.4, Math.min(88 + 180 * t, 255)) / 255 * 255;
     const g = Math.max(0.5, Math.min(50 + 160 * t, 255)) / 255 * 255;
     const b = Math.max(0.6, Math.min(120 - 40 * t, 255)) / 255 * 255;
     
     gradientStops.push(`rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`);
   }
   
   const gradient = `linear-gradient(to right, ${gradientStops.join(', ')})`;
   document.getElementById('colorbar-gradient').style.background = gradient;
 }


 // Function to toggle column settings panel
 function toggleColumnSettingsPanel() {
   const panel = document.getElementById('column-settings-panel');
   panel.classList.toggle('hidden');
 }

 /**
 * Populate Column Check Boxes 
 */
 function populateColumnCheckboxes(data) {
   // Get the column checkboxes container from the settings panel
   const columnCheckboxesContainer = document.querySelector('#column-settings-panel .checkbox-group');
   columnCheckboxesContainer.innerHTML = ''; // Clear existing checkboxes
   
   if (!data.points || data.points.length === 0) return;
   
   // Extract all available columns from the first point
   const allColumns = Object.keys(data.points[0]);
   
   // Initialize visibleColumns if it's empty - show first 4
   if (state.visibleColumns.length === 0) {
      // Just take the first 4 columns
      state.visibleColumns = allColumns.slice(0, 4);
   }
   
   // Create a checkbox for each column
   allColumns.forEach(column => {
       const checkboxDiv = document.createElement('div');
       checkboxDiv.className = 'column-checkbox';
       
       const checkbox = document.createElement('input');
       checkbox.type = 'checkbox';
       checkbox.id = `col-${column}`;
       checkbox.checked = state.visibleColumns.includes(column);
       checkbox.addEventListener('change', () => toggleColumnVisibility(column, checkbox.checked));
       
       const label = document.createElement('label');
       label.htmlFor = `col-${column}`;
       label.textContent = column;
       
       checkboxDiv.appendChild(checkbox);
       checkboxDiv.appendChild(label);
       columnCheckboxesContainer.appendChild(checkboxDiv);
   });
}
 

 // Function to toggle column visibility in table 
 function toggleColumnVisibility(column, isVisible) {
   if (isVisible && !state.visibleColumns.includes(column)) {
       state.visibleColumns.push(column);
   } else if (!isVisible) {
       state.visibleColumns = state.visibleColumns.filter(col => col !== column);
   }
   
   // Update the table if it's currently displayed
   if (state.selectedPoints.length > 0) {
       updateInfoBox();
   }
 }



 // ======== Data Loading Functions ========
 /**
  * Fetch and populate the JSON file list dropdown
  */
 function fetchJSONList() {
   // Show loading message
   showMessage("Fetching available JSON files...");
   
   // Fetch JSON file list from server endpoint
   fetch("/list_jsons")
       .then(response => {
           if (!response.ok) {
               throw new Error(`HTTP error! Status: ${response.status}`);
           }
           return response.json();
       })
       .then(files => {
           hideMessage();
           
           // Reset dropdown
           elements.jsonFileSelect.innerHTML = '<option value="">Select a JSON File</option>';
           
           if (files.length === 0) {
               showError(CONFIG.MESSAGES.NO_FILES);
               return;
           }
           
           // Populate dropdown
           files.forEach((file, index) => {
               const option = document.createElement("option");
               option.value = file;
               option.textContent = file;
               elements.jsonFileSelect.appendChild(option);
           });
           
           // Automatically select & load the first file
           if (files.length > 0) {
               elements.jsonFileSelect.selectedIndex = 1; // Select first file
               loadJSONData(files[0]);
           }
           
           console.log(`Found ${files.length} JSON files`);
       })
       .catch(error => {
           hideMessage();
           showError(`Error fetching JSON file list: ${error.message}`);
           console.error("Error fetching JSON list:", error);
           
           // Fallback to default files if fetch fails
           const fallbackFiles = CONFIG.DATA.DEFAULT_FILES;
           
           // Display warning
           showMessage("Using fallback file list");
           setTimeout(hideMessage, 3000);
           
           // Continue with fallback files
           populateDropdownWithFiles(fallbackFiles);
       });
       
   // Event listener for manual selection changes (only add once)
   if (!elements.jsonFileSelect._hasChangeListener) {
       elements.jsonFileSelect.addEventListener("change", function() {
           if (this.value) {
               loadJSONData(this.value);
           }
       });
       elements.jsonFileSelect._hasChangeListener = true;
   }
}

// Helper function to populate dropdown with files (used for fallback)
function populateDropdownWithFiles(files) {
   // Reset dropdown
   elements.jsonFileSelect.innerHTML = '<option value="">Select a JSON File</option>';
   
   if (files.length === 0) {
       showError(CONFIG.MESSAGES.NO_FILES);
       return;
   }
   
   // Populate dropdown
   files.forEach((file, index) => {
       const option = document.createElement("option");
       option.value = file;
       option.textContent = file;
       elements.jsonFileSelect.appendChild(option);
   });
   
   // Automatically select & load the default file
   const defaultIndex = CONFIG.DATA.DEFAULT_FILE_INDEX;
   if (defaultIndex >= 0 && defaultIndex < files.length) {
       elements.jsonFileSelect.selectedIndex = defaultIndex + 1; // +1 for the default empty option
       loadJSONData(files[defaultIndex]);
   }
}

 /**
  * Load JSON data from file
  * @param {string} filename - The file to load
  */
 function loadJSONData(filename) {
    // Show loading message
    showMessage(`Loading ${filename}...`);

    fetch(filename)
       .then(response => {
          if (!response.ok) {
             throw new Error(`HTTP error! Status: ${response.status}`);
          }
          return response.json();
       })
       .then(data => {
          // Hide loading message
          hideMessage();

          // Update visualization
          updateVisualization(data);
          console.log(`Loaded: ${filename}`);

          // Update dropdown selection to reflect the current file
          elements.jsonFileSelect.value = filename;

          // Populate the color selection dropdown
          populateColorDropdown(data);
       })
       .catch(error => {
          hideMessage();
          showError(`Error loading ${filename}: ${error.message}`);
          console.error("Error loading JSON:", error);
       });
 }

 /**
 * Reset the current selection
 */
function resetSelection() {
   console.log("Resetting selection");

   // Only proceed if we have a point cloud
   if (!state.pointCloud) {
      console.warn("No point cloud available");
      return;
   }

   const geometry = state.pointCloud.geometry;
   const colors = geometry.attributes.color.array;

   // Reset all points to current coloring scheme
   for (let i = 0; i < colors.length / 3; i++) {
      if (i < state.currentColors.length) {
         colors[i * 3] = state.currentColors[i][0];
         colors[i * 3 + 1] = state.currentColors[i][1];
         colors[i * 3 + 2] = state.currentColors[i][2];
      }
   }

   // Update color buffer
   geometry.attributes.color.needsUpdate = true;

   // Clear selected points array
   state.selectedPoints = [];

   // Update info display
   updateInfoBox();

   // Show feedback message
   showMessage("Selection cleared");
   setTimeout(hideMessage, 1500);
}

 /**
  * Update visualization with new data
  * @param {Object} newData - The new data to visualize
  */
 function updateVisualization(newData) {
    try {
       // Remove previous point cloud if it exists
       if (state.pointCloud) {
          state.scene.remove(state.pointCloud);
       }

       // Update points data
       state.points = newData.points || [];

       if (state.points.length === 0) {
          showError(CONFIG.MESSAGES.NO_POINTS);
          return;
       }

       // Create new point cloud
       createPointCloud();

       // Populate column checkboxes with the new data
       populateColumnCheckboxes(newData);

       // Start camera rotation for better overview
       rotateCameraAnimation(CONFIG.ANIMATION.DURATION);

       console.log(`Visualization updated with ${state.points.length} points`);
    } catch (error) {
       showError(`Failed to update visualization: ${error.message}`);
       console.error("Visualization update error:", error);
    }
 }

 /**
  * Populate color dropdown with available columns
  * @param {Object} data - The data object containing points
  */
 function populateColorDropdown(data) {
    elements.colorColumnSelect.innerHTML = '<option value="">Select Column</option>'; // Reset dropdown

    if (!data.points || data.points.length === 0) {
       showError(CONFIG.MESSAGES.NO_POINTS);
       return;
    }

    // Extract keys from the first point
    const columns = Object.keys(data.points[0]);

    // Populate the dropdown with column names
    columns.forEach(col => {
       const option = document.createElement("option");
       option.value = col;
       option.textContent = col;
       elements.colorColumnSelect.appendChild(option);
    });

    // Default to the 'x' column if available, otherwise use the first column
    if (columns.includes('x')) {
         elements.colorColumnSelect.value = 'x';
         updatePointColors('x'); // Apply initial coloring using 'x'
    } else if (columns.length > 0) {
         elements.colorColumnSelect.value = columns[0];
         updatePointColors(columns[0]); // Apply initial coloring with first column
   }

    // Add event listener to change coloring dynamically
    elements.colorColumnSelect.addEventListener("change", function() {
       if (this.value) {
          updatePointColors(this.value);
       }
    });
 }

 /**
  * Update point colors based on selected column
  * @param {string} column - The column to use for coloring
  */
 function updatePointColors(column) {
    if (!state.points || state.points.length === 0) {
       showError(CONFIG.MESSAGES.NO_POINTS);
       return;
    }

    // Find min/max values for the selected column
    let minVal = Infinity, maxVal = -Infinity;
    state.points.forEach(p => {
       if (p[column] !== undefined) {
          minVal = Math.min(minVal, p[column]);
          maxVal = Math.max(maxVal, p[column]);
       }
    });

    console.log(`Coloring points by ${column}: Min = ${minVal}, Max = ${maxVal}`);

    // Update colors in the geometry
    const geometry = state.pointCloud.geometry;
    const colors = geometry.attributes.color.array;
    state.currentColors = [];  // Reset stored colors

    state.points.forEach((point, i) => {
       if (point[column] !== undefined) {
          const value = point[column];
          const [r, g, b] = calculateColormap(value, minVal, maxVal);

          // Update buffer colors
          colors[i * 3] = r;
          colors[i * 3 + 1] = g;
          colors[i * 3 + 2] = b;

          // Store the current colors
          state.currentColors.push([r, g, b]);
       }
    });

    // Mark colors for update
    geometry.attributes.color.needsUpdate = true;

    // Reset any active selection
    state.selectedPoints = [];
    updateInfoBox();

   // Update Colorbar
    updateColorbar(column, minVal, maxVal);
 }

 // ======== Helper Functions ========
 /**
  * Show a message to the user
  * @param {string} message - The message to display
  */
 function showMessage(message) {
    elements.messageBox.textContent = message;
    elements.messageBox.style.display = "block";
 }

 /**
  * Hide the message box
  */
 function hideMessage() {
    elements.messageBox.style.display = "none";
 }

 /**
  * Show an error message
  * @param {string} message - The error message to display
  */
 function showError(message) {
    console.error(message);
    elements.errorBox.textContent = message;
    elements.errorBox.style.display = "block";

    // Hide error after 5 seconds
    setTimeout(() => {
       elements.errorBox.style.display = "none";
    }, 5000);
 }

/**
 * Create a circular texture for points
 * @returns {THREE.Texture} The circular texture
 */
function createCircleTexture() {
   const canvas = document.createElement('canvas');
   canvas.width = 64;
   canvas.height = 64;
   
   const context = canvas.getContext('2d');
   
   // Draw a circle
   context.beginPath();
   context.arc(32, 32, 32, 0, 2 * Math.PI);
   context.closePath();
   
   // Fill with white (will be colored by vertex colors)
   context.fillStyle = 'white';
   context.fill();
   
   const texture = new THREE.Texture(canvas);
   texture.needsUpdate = true;
   return texture;
}

 // Initialize when DOM is fully loaded
 document.addEventListener("DOMContentLoaded", init);
