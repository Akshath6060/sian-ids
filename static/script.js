// DOM Elements
const sequenceInput = document.getElementById('sequence');
const predictBtn = document.getElementById('predictBtn');
const exampleBtn = document.getElementById('exampleBtn');
const spinner = document.getElementById('spinner');
const resultsDiv = document.getElementById('results');
const limePlaceholder = document.getElementById('limePlaceholder');
const limeResults = document.getElementById('limeResults');

// Event Listeners
predictBtn.addEventListener('click', predict);
exampleBtn.addEventListener('click', loadExample);

// Load model info on page load
window.addEventListener('load', () => {
    loadModelInfo();
});

// Load example sequences
async function loadExample() {
    try {
        const response = await fetch('/api/example-sequences');
        const data = await response.json();
        sequenceInput.value = data.normal;
        sequenceInput.focus();
    } catch (error) {
        console.error('Error loading example:', error);
        alert('Error loading example sequences');
    }
}

// Make prediction
async function predict() {
    const sequence = sequenceInput.value.trim();
    
    if (!sequence) {
        alert('Please enter a syscall sequence');
        return;
    }

    showSpinner(true);
    resultsDiv.classList.add('hidden');
    limePlaceholder.classList.remove('hidden');
    limeResults.classList.add('hidden');

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ sequence: sequence })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Prediction failed');
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error: ' + error.message);
    } finally {
        showSpinner(false);
    }
}

// Display prediction results
function displayResults(data) {
    // Update prediction
    document.getElementById('predictionLabel').textContent = data.prediction;
    document.getElementById('confidence').textContent = data.confidence;
    
    // Update confidence bar
    const confidenceFill = document.getElementById('confidenceFill');
    confidenceFill.style.width = data.confidence + '%';
    
    // Color based on class
    const resultCard = document.getElementById('resultCard');
    if (data.class === 0) {
        resultCard.style.background = 'linear-gradient(135deg, #4CAF50 0%, #45a049 100%)';
    } else {
        resultCard.style.background = 'linear-gradient(135deg, #f44336 0%, #da190b 100%)';
    }

    // Display LIME features
    displayLimeFeatures(data.lime_features);

    // Show results
    resultsDiv.classList.remove('hidden');
    limePlaceholder.classList.add('hidden');
    limeResults.classList.remove('hidden');
}

// Display LIME features
function displayLimeFeatures(features) {
    const featuresList = document.getElementById('limeFeatures');
    featuresList.innerHTML = '';

    features.forEach(([feature, importance]) => {
        const item = document.createElement('div');
        item.className = 'feature-item';
        
        const name = document.createElement('span');
        name.className = 'feature-name';
        name.textContent = feature;
        
        const imp = document.createElement('span');
        imp.className = 'feature-importance';
        imp.textContent = importance.toFixed(4);
        
        item.appendChild(name);
        item.appendChild(imp);
        featuresList.appendChild(item);
    });
}

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch('/api/model-info');
        const data = await response.json();

        document.getElementById('accuracy').textContent = data.accuracy + '%';
        document.getElementById('precision').textContent = data.precision;
        document.getElementById('recall').textContent = data.recall;
        document.getElementById('f1score').textContent = data.f1_score;
        document.getElementById('features').textContent = data.features;
        document.getElementById('samples').textContent = data.total_samples;

        // Update confusion matrix
        const cm = data.confusion_matrix;
        const cmBody = document.getElementById('cmBody');
        cmBody.innerHTML = `
            <tr>
                <td>Normal</td>
                <td>${cm[0][0]}</td>
                <td>${cm[0][1]}</td>
            </tr>
            <tr>
                <td>Attack</td>
                <td>${cm[1][0]}</td>
                <td>${cm[1][1]}</td>
            </tr>
        `;

    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

// Show/hide spinner
function showSpinner(show) {
    if (show) {
        spinner.classList.remove('hidden');
    } else {
        spinner.classList.add('hidden');
    }
}
