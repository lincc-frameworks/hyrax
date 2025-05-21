/**
 * Image Viewer Module for displaying FITS images of selected points
 */

// Image viewer configuration
const IMAGE_CONFIG = {
    BATCH_SIZE: 6,          // Number of images to load at once
    MAX_IMAGES: 150,        // Maximum images to load in total
    FITS_EXTENSION: '.fits' // File extension for FITS images
};

// Image viewer state
const imageViewerState = {
    selectedIds: [],       // IDs of selected points
    loadedCount: 0,        // Number of images loaded so far
    isLoading: false,      // Flag to prevent concurrent loading
    fileNameColumn: '',    // Column name for image filenames
    filenameMap: {},       // Map of point IDs to image filenames
    cutoutsDir: 'cutouts'  // Default cutouts directory
};

// Cache DOM elements
const imageElements = {
    container: document.getElementById('image-viewer-container'),
    displayArea: document.getElementById('image-display-area'),
    loadMoreBtn: document.getElementById('load-more-images'),
    status: document.getElementById('image-status')
};

/**
 * Initialize the image viewer
 */
function initImageViewer() {
    console.log('Initializing image viewer module...');

    // Re-cache DOM elements in case they've changed
    imageElements.container = document.getElementById('image-viewer-container');
    imageElements.displayArea = document.getElementById('image-display-area');
    imageElements.loadMoreBtn = document.getElementById('load-more-images');
    imageElements.status = document.getElementById('image-status');
    
    // Check if elements exist
    if (!imageElements.container || !imageElements.displayArea) {
        console.error('Image viewer elements not found in the DOM');
        return;
    }
    
    // Add event listener for Load More button
    imageElements.loadMoreBtn.addEventListener('click', handleLoadMoreClick);
    
    // Fetch the cutouts directory from the server
    fetchCutoutsDirectory();
    
    console.log('Image viewer initialized');
}

/**
 * Fetch the cutouts directory configuration from the server
 */
function fetchCutoutsDirectory() {
    console.log('Fetching cutouts directory from server...');
    
    fetch("/get_cutouts_dir")
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            imageViewerState.cutoutsDir = data.cutouts_dir;
            console.log(`Cutouts directory set to: ${imageViewerState.cutoutsDir}`);
        })
        .catch(error => {
            console.error("Error fetching cutouts directory:", error);
            console.log("Using default cutouts directory: 'cutouts'");
            imageViewerState.cutoutsDir = 'cutouts';
        });
}

/**
 * Update the image viewer when selection changes
 * @param {Array} selectedPoints - Array of selected point IDs
 * @param {Array} points - Array of all point objects
 * @param {Array} columns - Array of available column names
 */
function updateImageViewer(selectedPoints, points, columns) {
    // Reset state for new selection
    imageViewerState.selectedIds = [...selectedPoints];
    imageViewerState.loadedCount = 0;
    imageViewerState.isLoading = false;
    
    // Clear display area
    imageElements.displayArea.innerHTML = '';
    
    // Find a suitable filename column (id, name, filename, etc.)
    determineFilenameColumn(columns);
    
    // Map point IDs to filenames if we have a filename column
    if (imageViewerState.fileNameColumn) {
        mapIdsToFilenames(selectedPoints, points);
    }
    
    // Show or hide the viewer based on selection state
    if (selectedPoints.length > 0 && imageViewerState.fileNameColumn) {
        // Show the container
        imageElements.container.style.display = 'block';
        
        // Load the first batch of images
        loadImageBatch(0, IMAGE_CONFIG.BATCH_SIZE);
        
        // Update load more button state
        updateLoadMoreButton();
        
        // Update status
        updateImageStatus();
    } else {
        // Hide the container
        imageElements.container.style.display = 'none';
    }
}

/**
 * Determine which column to use for image filenames
 * @param {Array} columns - Available column names
 */
function determineFilenameColumn(columns) {
    // Priority list of column names that could contain filenames
    const fileNamePriorities = ['filename', 'filenames'];
    
    // Find the first matching column
    imageViewerState.fileNameColumn = '';
    
    for (const priority of fileNamePriorities) {
        const match = columns.find(col => 
            col.toLowerCase().includes(priority)
        );
        
        if (match) {
            imageViewerState.fileNameColumn = match;
            console.log(`Using column "${match}" for image filenames`);
            return;
        }
    }
    
    // If no suitable column found, explicitly check for 'id' column
    // but warn that it might not be appropriate for filenames
    if (columns.includes('id')) {
        imageViewerState.fileNameColumn = 'id';
        console.warn('Using "id" for image filenames - this may not be appropriate if IDs are not actual filenames');
        return;
    }
    
    console.warn('No suitable column found for image filenames');
}

/**
 * Map point IDs to filenames
 * @param {Array} selectedIds - Selected point IDs
 * @param {Array} points - All point objects
 */
function mapIdsToFilenames(selectedIds, points) {
    imageViewerState.filenameMap = {};
    
    // Get the full point objects for selected IDs
    const selectedPoints = points.filter(point => 
        selectedIds.includes(point.id)
    );
    
    // Create mapping from ID to filename
    selectedPoints.forEach(point => {
        const id = point.id;
        let filename = point[imageViewerState.fileNameColumn];
        
        // If the filename doesn't have the .fits extension, add it
        if (!filename.endsWith(IMAGE_CONFIG.FITS_EXTENSION)) {
            filename = `${filename}${IMAGE_CONFIG.FITS_EXTENSION}`;
        }
        
        imageViewerState.filenameMap[id] = filename;
    });
}

/**
 * Load a batch of images
 * @param {number} startIndex - Starting index to load from
 * @param {number} count - Number of images to load
 */
function loadImageBatch(startIndex, count) {
    if (imageViewerState.isLoading) return;
    
    imageViewerState.isLoading = true;
    
    // Get the IDs to load in this batch
    const batchIds = imageViewerState.selectedIds.slice(
        startIndex, 
        startIndex + count
    );
    
    if (batchIds.length === 0) {
        imageViewerState.isLoading = false;
        return;
    }
    
    console.log(`Loading image batch: ${startIndex} to ${startIndex + batchIds.length - 1}`);
    
    // NEW: Track if container is scrollable before loading
    const wasScrollable = isScrollable(imageElements.displayArea);
    
    // Create placeholders for each image
    batchIds.forEach(id => {
        createImagePlaceholder(id);
    });
    
    // Load each image in the batch
    const loadPromises = batchIds.map(id => loadFITSImage(id));
    
    // Update state when all images in the batch are loaded
    Promise.allSettled(loadPromises).then(() => {
        imageViewerState.loadedCount += batchIds.length;
        imageViewerState.isLoading = false;
        updateLoadMoreButton();
        updateImageStatus();
        
        // NEW: Check if scrollbar just appeared and auto-scroll
        const isNowScrollable = isScrollable(imageElements.displayArea);
        if (!wasScrollable && isNowScrollable) {
            // Add a small delay to let images render
            setTimeout(() => {
                const firstNewImage = document.getElementById(`image-item-${batchIds[0]}`);
                if (firstNewImage) {
                    firstNewImage.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }, 300);
        }
    });
}

/**
 * Create a placeholder for an image while it loads
 * @param {string|number} id - Point ID
 */
function createImagePlaceholder(id) {
    const placeholder = document.createElement('div');
    placeholder.className = 'image-item';
    placeholder.id = `image-item-${id}`;
    
    // Add title
    const title = document.createElement('div');
    title.className = 'image-title';
    title.textContent = `ID: ${id}`;
    placeholder.appendChild(title);
    
    // Add loading indicator with debug info
    const loading = document.createElement('div');
    loading.className = 'image-loading';
    loading.setAttribute('data-load-start', Date.now().toString());
    placeholder.appendChild(loading);
    
    // Add to display area
    imageElements.displayArea.appendChild(placeholder);
}

/**
 * Load a FITS image for a given point ID
 * @param {string|number} id - Point ID
 * @returns {Promise} - Promise that resolves when the image is loaded
 */
function loadFITSImage(id) {
    return new Promise((resolve, reject) => {
        const filename = imageViewerState.filenameMap[id];
        
        if (!filename) {
            handleImageError(id, 'Missing filename');
            resolve();
            return;
        }
        
        // Joining the filepaths properly
        const filepath = imageViewerState.cutoutsDir.endsWith('/') 
            ? `${imageViewerState.cutoutsDir}${filename}`  
            : `${imageViewerState.cutoutsDir}/${filename}`;  
        
        console.log(`DEBUG: Loading FITS file from: ${filepath}`);
        
        // Set a timeout for this image
        const timeoutId = setTimeout(() => {
            console.error(`DEBUG: Timeout loading image ${id}`);
            handleImageError(id, 'Timeout loading image');
            resolve();
        }, 30000); // 30 second timeout
        
        // Fetch the FITS file
        fetch(filepath)
            .then(response => {
                //console.log(`DEBUG: Fetch response status: ${response.status}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.arrayBuffer();
            })
            .then(buffer => {
                //console.log(`DEBUG: Received buffer, size: ${buffer.byteLength}`);
                // Process the FITS file
                return processFITSFile(buffer);
            })
            .then(imageData => {
                clearTimeout(timeoutId);
                //console.log('DEBUG: Successfully processed FITS data');
                // Display the processed image
                displayProcessedImage(id, imageData);
                resolve();
            })
            .catch(error => {
                clearTimeout(timeoutId);
                console.error(`Error loading image for ID ${id}:`, error);
                console.error(`Full error stack:`, error.stack);
                handleImageError(id, error.message);
                resolve(); // Resolve anyway to continue with batch
            });
    });
}

/**
 * Process a FITS file buffer into image data
 * @param {ArrayBuffer} buffer - FITS file buffer
 * @returns {Object} - Processed image data
 */
function processFITSFile(buffer) {
    return new Promise((resolve, reject) => {
        try {
            //console.log(`DEBUG: Starting FITS processing. Buffer size: ${buffer.byteLength}`);
            
            // Convert ArrayBuffer to Blob
            const blob = new Blob([buffer], { type: 'application/fits' });
            //console.log('DEBUG: Created blob from buffer');
            
            // Parse the FITS file using fitsjs
            const fits = new FITS(blob, function() {
                try {
                    //console.log('DEBUG: FITS file parsed successfully');
                    
                    // Get the first HDU (Header Data Unit)
                    const hdu = fits.getHDU();
                    
                    // The dimensions are in hdu.data.width and hdu.data.height
                    const width = hdu.data.width;
                    const height = hdu.data.height;
                    //console.log(`DEBUG: HDU dimensions: ${width}x${height}`);
                    
                    // Correct way to get pixel data from fits.js library
                    hdu.data.getFrame(0, function(frame) {
                        try {
                            //console.log(`DEBUG: Frame received, length: ${frame.length}`);
                            //console.log(`DEBUG: Frame type: ${frame.constructor.name}`);
                            //console.log(`DEBUG: First 5 raw values:`, frame.slice(0, 5));
                            
                            // Check if frame is empty or has issues
                            if (!frame || frame.length === 0) {
                                throw new Error('Empty frame received');
                            }
                            
                            // Check for NaN or infinite values
                            let hasInvalidValues = false;
                            let invalidCount = 0;
                            for (let i = 0; i < frame.length; i++) {
                                if (!isFinite(frame[i])) {
                                    hasInvalidValues = true;
                                    invalidCount++;
                                }
                            }
                            //console.log(`DEBUG: Invalid values count: ${invalidCount} of ${frame.length}`);
                            
                            // Process the image data
                            const processedData = processImageData(frame, width, height);
                            //console.log('DEBUG: Image data processed successfully');
                            
                            resolve({
                                width,
                                height,
                                data: processedData
                            });
                        } catch (error) {
                            console.error('DEBUG: Error in getFrame callback:', error);
                            reject(error);
                        }
                    });
                    
                } catch (error) {
                    console.error('DEBUG: Error processing HDU:', error);
                    reject(error);
                }
            });
            
        } catch (error) {
            console.error('DEBUG: Error initializing FITS reader:', error);
            reject(error);
        }
    });
}

/**
 * Process raw image data (normalize and log transform)
 * @param {TypedArray} data - Raw image data
 * @param {number} width - Image width
 * @param {number} height - Image height
 * @returns {Uint8ClampedArray} - Processed image data for display
 */
function processImageData(data, width, height) {
    try {
        // Create new array for processed data
        const processedData = new Uint8ClampedArray(width * height * 4); // RGBA
        
        // Find min and max values for normalization
        let logMin = Infinity;
        let logMax = -Infinity;
        let validCount = 0;
        const shift = 1e-3;
     
        // First pass: find valid min/max values and count valid pixels
        for (let i = 0; i < data.length; i++) {
            const value = data[i];
            if (value !== null && value !== undefined && isFinite(value) && value > 0) {
                const logValue = Math.log10(value + shift);
                validCount++;
                if (logValue < logMin) logMin = logValue;
                if (logValue > logMax) logMax = logValue;
            }
        }
        
        //console.log(`DEBUG: Valid pixels: ${validCount} of ${data.length}`);
        
        // If no valid values found or all values are the same, use defaults
        if (!isFinite(logMin) || !isFinite(logMax) || logMin === logMax) {
            console.warn('DEBUG: Invalid or flat log scale range, using defaults');
            logMin = 0;
            logMax = 1;
        }
        
        //console.log(`DEBUG: Raw data stats - Min: ${min}, Max: ${max}`);
        
        // Normalize and transform
        for (let i = 0; i < data.length; i++) {
            const value = data[i];
            
            // Calculate pixel RGBA values
            let pixelValue;
            
            if (value !== null && value !== undefined && isFinite(value) && value > 0) {

                // Applying a log-norm scaling
                const logValue = Math.log10(value + shift);
                const normalized = (logValue - logMin) / (logMax - logMin);

                // Now let's soften the scaling a bit with a tunable gamma
                const gamma = 0.9;  // Try values between 0.4 and 0.9
                const compressed = Math.pow(normalized, gamma);

                // Convert to 0-255 range
                pixelValue = Math.floor(compressed * 255);

                // Clamp to [0, 255]
                if (pixelValue < 0) pixelValue = 0;
                if (pixelValue > 255) pixelValue = 255;
                
                // Safety check
                if (!isFinite(pixelValue) || pixelValue < 0 || pixelValue > 255) {
                    console.warn(`DEBUG: Invalid pixel value: ${pixelValue}`);
                    pixelValue = 0;
                }
            } else {
                pixelValue = 0; // Set to black for invalid values
            }
            
            // Set RGBA values (grayscale)
            const idx = i * 4;
            processedData[idx] = pixelValue;     // R
            processedData[idx + 1] = pixelValue; // G
            processedData[idx + 2] = pixelValue; // B
            processedData[idx + 3] = 255;        // A (opaque)
        }
        
        // Check if the processed data has any non-black pixels
        let hasNonBlackPixels = false;
        for (let i = 0; i < processedData.length; i += 4) {
            if (processedData[i] > 0 || processedData[i+1] > 0 || processedData[i+2] > 0) {
                hasNonBlackPixels = true;
                break;
            }
        }
        //console.log(`DEBUG: Has non-black pixels: ${hasNonBlackPixels}`);
        
        return processedData;
    } catch (error) {
        console.error('DEBUG: Error in processImageData:', error);
        throw error;
    }
}

/**
 * Display a processed image
 * @param {string|number} id - Point ID
 * @param {Object} imageData - Processed image data
 */
 function displayProcessedImage(id, imageData) {
    const itemContainer = document.getElementById(`image-item-${id}`);
    
    if (!itemContainer) {
        console.error(`DEBUG: Container not found for ID ${id}`);
        return;
    }
    
    // Remove loading indicator
    const loadingIndicator = itemContainer.querySelector('.image-loading');
    if (loadingIndicator) {
        itemContainer.removeChild(loadingIndicator);
    }
    
    // Create canvas for image display
    const canvas = document.createElement('canvas');
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    
    //console.log(`DEBUG: Canvas size: ${canvas.width}x${canvas.height}`);
    //console.log(`DEBUG: Image data size: ${imageData.data.length}`);
    
    // Get canvas context and draw image
    const ctx = canvas.getContext('2d');
    const imgData = new ImageData(
        imageData.data, 
        imageData.width, 
        imageData.height
    );
    
    //console.log(`DEBUG: ImageData created successfully`);
    
    ctx.putImageData(imgData, 0, 0);
    
    // Verify the canvas is being displayed
    //console.log(`DEBUG: Canvas added to container for ID ${id}`);
    
    // Add canvas to container
    itemContainer.appendChild(canvas);
    
    // Force a repaint
    requestAnimationFrame(() => {
        const renderedCanvas = itemContainer.querySelector('canvas');
        //console.log(`DEBUG: Canvas in DOM: ${renderedCanvas ? 'Yes' : 'No'}`);
    });
}

/**
 * Handle image loading errors
 * @param {string|number} id - Point ID
 * @param {string} errorMessage - Error message
 */
function handleImageError(id, errorMessage) {
    const itemContainer = document.getElementById(`image-item-${id}`);
    
    if (!itemContainer) return;
    
    // Calculate how long it took to fail
    const loadingIndicator = itemContainer.querySelector('.image-loading');
    if (loadingIndicator) {
        const loadStart = loadingIndicator.getAttribute('data-load-start');
        if (loadStart) {
            const duration = Date.now() - parseInt(loadStart);
            console.log(`DEBUG: Image ${id} failed after ${duration}ms`);
        }
        itemContainer.removeChild(loadingIndicator);
    }
    
    // Update title to show error
    const title = itemContainer.querySelector('.image-title');
    if (title) {
        title.textContent = `Error: ${id}`;
        title.style.color = '#ef4444'; // Red color for errors
    }
    
    // Add error message
    const errorElement = document.createElement('div');
    errorElement.className = 'image-error';
    errorElement.textContent = errorMessage.substring(0, 50); // Truncate long messages
    errorElement.title = errorMessage; // Full message on hover
    itemContainer.appendChild(errorElement);
    
    console.error(`DEBUG: Image ${id} error: ${errorMessage}`);
}

/**
 * Handle click on Load More button
 */
function handleLoadMoreClick() {
    if (imageViewerState.isLoading) return;
    
    // Load the next batch of images
    loadImageBatch(
        imageViewerState.loadedCount,
        IMAGE_CONFIG.BATCH_SIZE
    );
}

/**
 * Update the state of the Load More button
 */
function updateLoadMoreButton() {
    const totalCount = imageViewerState.selectedIds.length;
    const loadedCount = imageViewerState.loadedCount;
    
    // Enable/disable the button based on whether there are more images to load
    if (loadedCount >= totalCount || loadedCount >= IMAGE_CONFIG.MAX_IMAGES) {
        imageElements.loadMoreBtn.classList.add('disabled');
        imageElements.loadMoreBtn.disabled = true;
    } else {
        imageElements.loadMoreBtn.classList.remove('disabled');
        imageElements.loadMoreBtn.disabled = false;
    }
}

/**
 * Update the image status text
 */
function updateImageStatus() {
    const totalCount = imageViewerState.selectedIds.length;
    const loadedCount = Math.min(
        imageViewerState.loadedCount,
        totalCount,
        IMAGE_CONFIG.MAX_IMAGES
    );
    
    imageElements.status.textContent = 
        `Showing ${loadedCount} of ${totalCount} images`;
}

// Initialize the image viewer when the script loads
document.addEventListener('DOMContentLoaded', function() {
    // Initialize only after main script has initialized
    if (typeof state !== 'undefined') {
        initImageViewer();
    } else {
        // If main script isn't loaded yet, wait for it
        window.addEventListener('load', initImageViewer);
    }
});