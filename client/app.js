// Modern Celebrity Face Recognition App

const APP = {
    API_URL: 'https://celeb-face-rec-backend.onrender.com',
    currentFile: null,
    celebrities: [], // Will be loaded from API
    
    // Initialize the application
    init() {
        this.loadCelebrities();
        this.setupEventHandlers();
        this.showEmptyState();
    },
    
    // Load celebrities from API or use defaults
    async loadCelebrities() {
        try {
            const response = await fetch(`${this.API_URL}/classes`);
            if (response.ok) {
                const data = await response.json();
                if (data.success && data.classes) {
                    this.celebrities = data.classes.map(name => ({
                        id: name.toLowerCase().replace(' ', '_'),
                        name: name,
                        image: `./images/${name.toLowerCase().split(' ')[0]}.jpg`
                    }));
                }
            }
        } catch (error) {
            console.log('Using default celebrities');
            // Default celebrities if API fails
            this.celebrities = [
                { id: 'keanu_reeves', name: 'Keanu Reeves', image: './images/keanu.jpg' },
                { id: 'margot_robbie', name: 'Margot Robbie', image: './images/margot.jpg' },
                { id: 'scarlett_johansson', name: 'Scarlett Johansson', image: './images/scarlett.jpg' },
                { id: 'zendaya', name: 'Zendaya', image: './images/zendaya.jpg' },
                { id: 'will_smith', name: 'Will Smith', image: './images/will.jpg' }
            ];
        }
        
        this.renderCelebrityGrid();
    },
    
    // Render celebrity grid
    renderCelebrityGrid() {
        const grid = document.getElementById('celebrityGrid');
        grid.innerHTML = '';
        
        this.celebrities.forEach(celeb => {
            const card = document.createElement('div');
            card.className = 'celebrity-card';
            card.dataset.celebrity = celeb.id;
            card.innerHTML = `
                <div class="celebrity-avatar">
                    <img src="${celeb.image}" alt="${celeb.name}" onerror="this.src='./images/default-avatar.png'">
                </div>
                <p class="celebrity-name">${celeb.name}</p>
            `;
            grid.appendChild(card);
        });
    },
    
    // Setup all event handlers
    setupEventHandlers() {
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const classifyBtn = document.getElementById('classifyBtn');
        const removeBtn = document.getElementById('removeImage');
        const clearBtn = document.getElementById('clearAllBtn');
        
        // File input
        dropZone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleFiles(e.target.files));
        
        // Drag and drop
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            this.handleFiles(e.dataTransfer.files);
        });
        
        // Buttons
        classifyBtn.addEventListener('click', () => this.classifyImage());
        removeBtn.addEventListener('click', () => this.clearImage());
        clearBtn.addEventListener('click', () => this.clearAll());
        
        // Prevent default drag behavior on window
        window.addEventListener('dragover', (e) => e.preventDefault());
        window.addEventListener('drop', (e) => e.preventDefault());
    },
    
    // Handle file selection
    handleFiles(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        
        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please upload a valid image file (JPG, PNG, GIF)');
            return;
        }
        
        // Validate file size (10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showError('Image size must be less than 10MB');
            return;
        }
        
        this.currentFile = file;
        this.displayImage(file);
    },
    
    // Display selected image
    displayImage(file) {
        const reader = new FileReader();
        
        reader.onload = (e) => {
            const previewImg = document.getElementById('previewImg');
            previewImg.src = e.target.result;
            
            document.getElementById('dropZone').classList.add('hidden');
            document.getElementById('imagePreview').classList.remove('hidden');
            
            this.showEmptyState();
        };
        
        reader.onerror = () => {
            this.showError('Failed to read image file');
        };
        
        reader.readAsDataURL(file);
    },
    
    // Classify the image
    async classifyImage() {
        if (!this.currentFile) return;
        
        const classifyBtn = document.getElementById('classifyBtn');
        classifyBtn.disabled = true;
        
        // Show loading state
        this.hideAllSections();
        document.getElementById('loadingState').classList.remove('hidden');
        
        try {
            // Read file as base64
            const base64Data = await this.fileToBase64(this.currentFile);
            
            // Send to API
            const response = await fetch(`${this.API_URL}/classify_image`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `image_data=${encodeURIComponent(base64Data)}`
            });
            
            const data = await response.json();
            
            if (response.ok && data.success) {
                if (data.faces_detected > 0) {
                    this.displayResults(data.results, data.faces_detected);
                    document.getElementById('clearAllBtn').classList.remove('hidden');
                } else {
                    this.showError(data.message || 'No faces detected in the image');
                }
            } else {
                this.showError(data.error || 'Failed to process image');
            }
        } catch (error) {
            console.error('Classification error:', error);
            this.showError('Network error. Please check your connection and try again.');
        } finally {
            classifyBtn.disabled = false;
            document.getElementById('loadingState').classList.add('hidden');
        }
    },
    
    // Convert file to base64
    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    },
    
    // Display classification results
    displayResults(results, totalFaces) {
        this.hideAllSections();
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.innerHTML = '';
        resultsSection.classList.remove('hidden');
        
        results.forEach((result, index) => {
            const card = this.createResultCard(result, index + 1, totalFaces);
            resultsSection.appendChild(card);
        });
    },
    
    // Create a result card for a single face
    createResultCard(result, faceNumber, totalFaces) {
        const card = document.createElement('div');
        card.className = 'result-card glass-effect';
        
        // Determine confidence level
        let confidenceClass = 'confidence-low';
        if (result.confidence >= 80) {
            confidenceClass = 'confidence-high';
        } else if (result.confidence >= 60) {
            confidenceClass = 'confidence-medium';
        }
        
        card.innerHTML = `
            <div class="result-header">
                <h3 class="result-title">
                    <span class="face-indicator">${faceNumber}</span>
                    Face ${faceNumber} of ${totalFaces}
                </h3>
                <span class="confidence-badge ${confidenceClass}">
                    ${result.confidence}% confident
                </span>
            </div>
            
            <div class="predicted-celebrity">
                <p class="predicted-label">Identified as:</p>
                <p class="predicted-name">${result.predicted_class}</p>
            </div>
            
            <div class="probability-section">
                <p class="probability-title">Top Matches:</p>
                <div class="probability-list">
                    ${result.top_3.map(prob => this.createProbabilityBar(prob)).join('')}
                </div>
            </div>
        `;
        
        return card;
    },
    
    // Create probability bar HTML
    createProbabilityBar(prob) {
        let barClass = 'probability-bar-low';
        if (prob.probability >= 50) {
            barClass = 'probability-bar-high';
        } else if (prob.probability >= 25) {
            barClass = 'probability-bar-medium';
        }
        
        return `
            <div class="probability-item">
                <span class="probability-name">${prob.name}</span>
                <div class="probability-bar-container">
                    <div class="probability-bar ${barClass}" style="width: ${prob.probability}%"></div>
                    <span class="probability-value">${prob.probability}%</span>
                </div>
            </div>
        `;
    },
    
    // Show error message
    showError(message) {
        this.hideAllSections();
        document.getElementById('errorText').textContent = message;
        document.getElementById('errorMessage').classList.remove('hidden');
    },
    
    // Show empty state
    showEmptyState() {
        this.hideAllSections();
        document.getElementById('emptyState').classList.remove('hidden');
    },
    
    // Hide all result sections
    hideAllSections() {
        ['resultsSection', 'errorMessage', 'emptyState', 'loadingState'].forEach(id => {
            document.getElementById(id).classList.add('hidden');
        });
    },
    
    // Clear selected image
    clearImage() {
        this.currentFile = null;
        document.getElementById('fileInput').value = '';
        document.getElementById('imagePreview').classList.add('hidden');
        document.getElementById('dropZone').classList.remove('hidden');
        this.showEmptyState();
        document.getElementById('clearAllBtn').classList.add('hidden');
    },
    
    // Clear everything
    clearAll() {
        this.clearImage();
        this.showEmptyState();
    }
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    APP.init();
});