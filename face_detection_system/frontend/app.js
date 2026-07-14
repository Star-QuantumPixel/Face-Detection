const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const toggleBtn = document.getElementById('toggle-btn');
const statusOverlay = document.getElementById('status-overlay');
const faceCountEl = document.getElementById('face-count');
const latencyEl = document.getElementById('latency-val');
const fpsSlider = document.getElementById('fps-slider');
const fpsVal = document.getElementById('fps-val');

const enrollNameInput = document.getElementById('enroll-name');
const enrollBtn = document.getElementById('enroll-btn');
const enrollMsg = document.getElementById('enroll-msg');
const toastContainer = document.getElementById('toast-container');

let isDetecting = false;
let detectionLoopId = null;
let targetFps = 15;

// Configurable API URLs
const API_URL = 'http://localhost:8080/detect';
const ENROLL_URL = 'http://localhost:8080/enroll';

// Update FPS display
fpsSlider.addEventListener('input', (e) => {
    targetFps = parseInt(e.target.value, 10);
    fpsVal.textContent = targetFps;
});

// Setup Camera
async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "user"
            }
        });
        video.srcObject = stream;
        
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                video.classList.add('ready');
                document.getElementById('skeleton-overlay').classList.add('hidden');
                resolve(video);
            };
        });
    } catch (err) {
        console.error("Error accessing camera:", err);
        statusOverlay.innerHTML = `<p style="color:#ef4444">Camera access denied.</p>`;
        throw err;
    }
}

// Ensure canvas matches video size
function resizeCanvas() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
}

// Convert video frame to base64
function getFrameB64() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    
    const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.8);
    return dataUrl.split(',')[1];
}

// Animation State
let targetDetections = [];
let currentDetections = [];
const LERP_FACTOR = 0.25;

// Render Loop for smooth interpolation
function renderLoop() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!isDetecting) return;

    // Interpolate
    for (let i = 0; i < targetDetections.length; i++) {
        const target = targetDetections[i];
        
        if (!currentDetections[i]) {
            currentDetections[i] = { ...target };
        } else {
            currentDetections[i].x += (target.x - currentDetections[i].x) * LERP_FACTOR;
            currentDetections[i].y += (target.y - currentDetections[i].y) * LERP_FACTOR;
            currentDetections[i].w += (target.w - currentDetections[i].w) * LERP_FACTOR;
            currentDetections[i].h += (target.h - currentDetections[i].h) * LERP_FACTOR;
            currentDetections[i].confidence = target.confidence;
            currentDetections[i].label = target.label;
        }
    }
    
    // Trim excess
    if (currentDetections.length > targetDetections.length) {
        currentDetections.length = targetDetections.length;
    }
    
    // Draw
    currentDetections.forEach(det => {
        const { x, y, w, h, confidence, label } = det;
        const flippedX = canvas.width - (x + w);
        
        // Cyberpunk Glow
        ctx.shadowColor = 'rgba(16, 185, 129, 0.6)';
        ctx.shadowBlur = 12;
        
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2.5;
        
        // Smooth rounded rect for bounding box
        ctx.beginPath();
        ctx.roundRect(flippedX, y, w, h, 12);
        ctx.stroke();
        
        ctx.shadowBlur = 0; // Turn off for text
        
        ctx.fillStyle = '#10b981';
        const text = `${label} ${(confidence * 100).toFixed(0)}%`;
        ctx.font = '600 13px Outfit, sans-serif';
        const textWidth = ctx.measureText(text).width;
        
        ctx.beginPath();
        ctx.roundRect(flippedX, y - 28, textWidth + 20, 28, [12, 12, 0, 0]);
        ctx.fill();
        
        ctx.fillStyle = '#050505';
        ctx.fillText(text, flippedX + 10, y - 9);
    });
    
    requestAnimationFrame(renderLoop);
}

// Main detection loop
async function detectFrame() {
    if (!isDetecting) return;
    
    const startTime = performance.now();
    
    try {
        const b64 = getFrameB64();
        
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image_b64: b64 })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        targetDetections = result.detections;
        faceCountEl.textContent = result.face_count;
        latencyEl.textContent = `${result.latency_ms.toFixed(1)} ms`;
        
    } catch (err) {
        console.error("Detection error:", err);
        showToast("Connection to server lost. Is the API running?", "error");
        if (isDetecting) toggleBtn.click(); // Stop detection automatically
        return;
    }
    
    const elapsed = performance.now() - startTime;
    const targetDelay = 1000 / targetFps;
    const delay = Math.max(0, targetDelay - elapsed);
    
    detectionLoopId = setTimeout(detectFrame, delay);
}

// Toggle detection
toggleBtn.addEventListener('click', async () => {
    if (isDetecting) {
        // Stop
        isDetecting = false;
        clearTimeout(detectionLoopId);
        targetDetections = [];
        currentDetections = [];
        toggleBtn.textContent = 'Start Detection';
        toggleBtn.classList.remove('active');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        faceCountEl.textContent = '0';
        latencyEl.textContent = '-- ms';
    } else {
        // Start
        if (!video.srcObject) {
            await init();
        }
        isDetecting = true;
        toggleBtn.textContent = 'Stop Detection';
        toggleBtn.classList.add('active');
        detectFrame();
        requestAnimationFrame(renderLoop);
    }
});

// Init
async function init() {
    try {
        await setupCamera();
        resizeCanvas();
        statusOverlay.classList.add('hidden');
    } catch (err) {
        // Handle error visually if needed
    }
}

// Enrollment logic
enrollBtn.addEventListener('click', async () => {
    const name = enrollNameInput.value.trim();
    if (!name) {
        showToast('Please enter a name.', 'error');
        return;
    }
    if (!video.srcObject) {
        showToast('Please start detection first.', 'error');
        return;
    }

    try {
        enrollBtn.disabled = true;
        enrollBtn.textContent = 'Enrolling...';
        
        const b64 = getFrameB64();
        
        const response = await fetch(ENROLL_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image_b64: b64, name: name })
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || `Error ${response.status}`);
        }
        
        showToast(result.message || `Successfully enrolled ${name}`, 'success');
        enrollNameInput.value = '';
    } catch (err) {
        console.error("Enrollment error:", err);
        showToast(err.message, 'error');
    } finally {
        enrollBtn.disabled = false;
        enrollBtn.textContent = 'Enroll';
    }
});

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${message}</span>`;
    toastContainer.appendChild(toast);
    
    // Animate in
    setTimeout(() => toast.classList.add('show'), 10);
    
    // Remove after 4s
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Start camera on page load
init();
