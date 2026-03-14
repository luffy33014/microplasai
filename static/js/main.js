document.addEventListener('DOMContentLoaded', function () {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const cancelBtn = document.getElementById('cancel-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const uploadForm = document.getElementById('upload-form');
    const loadingSpinner = document.getElementById('loading-spinner');

    // Camera elements
    const startCameraBtn = document.getElementById('start-camera-btn');
    const stopCameraBtn = document.getElementById('stop-camera-btn');
    const snapBtn = document.getElementById('snap-btn');
    const cameraContainer = document.getElementById('camera-container');
    const videoElement = document.getElementById('microscope-video');
    const captureCanvas = document.getElementById('capture-canvas');
    const cameraLoading = document.getElementById('camera-loading');

    let currentStream = null;

    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFiles);

    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        uploadArea.classList.add('drag-active');
    }

    function unhighlight(e) {
        uploadArea.classList.remove('drag-active');
    }

    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        handleFiles();
    }

    function handleFiles() {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function (e) {
                    imagePreview.src = e.target.result;
                    uploadArea.classList.add('hidden');
                    previewContainer.classList.remove('hidden');
                };
                reader.readAsDataURL(file);
            } else {
                alert('Please upload an image file.');
            }
        }
    }

    // Cancel upload
    cancelBtn.addEventListener('click', () => {
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        uploadArea.classList.remove('hidden');
    });

    // Camera functionality
    startCameraBtn.addEventListener('click', async () => {
        uploadArea.classList.add('hidden');
        cameraContainer.classList.remove('hidden');
        cameraLoading.classList.remove('hidden');
        videoElement.classList.add('hidden');

        try {
            // Request camera access (facingMode environment usually targets rear/USB cameras)
            currentStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1920 }, height: { ideal: 1080 } }
            });
            videoElement.srcObject = currentStream;
            videoElement.onloadedmetadata = () => {
                cameraLoading.classList.add('hidden');
                videoElement.classList.remove('hidden');
            };
        } catch (err) {
            console.error("Error accessing the camera: ", err);
            cameraLoading.innerHTML = `<p class="text-danger">Failed to access camera. Please ensure it's connected and permissions are granted.</p>`;
        }
    });

    function stopCamera() {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
        }
        videoElement.srcObject = null;
    }

    stopCameraBtn.addEventListener('click', () => {
        stopCamera();
        cameraContainer.classList.add('hidden');
        uploadArea.classList.remove('hidden');
        // reset UI
        cameraLoading.classList.remove('hidden');
        cameraLoading.innerHTML = `<div class="spinner"></div><p>Accessing Microscope...</p>`;
    });

    snapBtn.addEventListener('click', () => {
        if (!currentStream) return;

        // Ensure canvas matches video aspect ratio and resolution
        captureCanvas.width = videoElement.videoWidth;
        captureCanvas.height = videoElement.videoHeight;

        const context = captureCanvas.getContext('2d');
        context.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);

        // Stop the camera since we got the image
        stopCamera();

        // Show preview container
        cameraContainer.classList.add('hidden');
        previewContainer.classList.remove('hidden');

        // Set image preview from canvas
        const dataUrl = captureCanvas.toDataURL('image/png');
        imagePreview.src = dataUrl;

        // Convert dataURL to Blob and create a File object to insert into the input[type=file]
        captureCanvas.toBlob((blob) => {
            const file = new File([blob], "microscope_capture_" + Date.now() + ".png", { type: "image/png" });
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
        }, 'image/png');
    });

    // Analyze button
    analyzeBtn.addEventListener('click', () => {
        if (fileInput.files.length > 0) {
            loadingSpinner.classList.remove('hidden');
            analyzeBtn.disabled = true;
            cancelBtn.disabled = true;
            uploadForm.submit();
        }
    });
});
