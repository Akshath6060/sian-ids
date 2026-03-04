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
    loadGraphs();
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

// Chart instances
let trainingHistoryChart = null;
let rocCurveChart = null;
let prCurveChart = null;

// Load graphs
async function loadGraphs() {
    try {
        await loadTrainingHistory();
        await loadROCCurve();
        await loadPrecisionRecallCurve();
    } catch (error) {
        console.error('Error loading graphs:', error);
    }
}

// Load training history graph
async function loadTrainingHistory() {
    try {
        const response = await fetch('/api/graphs/training-history');
        const data = await response.json();
        
        if (!data.loss || data.loss.length === 0) {
            console.log('No training history available yet');
            return;
        }

        const ctx = document.getElementById('trainingHistoryChart');
        const epochs = Array.from({length: data.loss.length}, (_, i) => i + 1);

        if (trainingHistoryChart) {
            trainingHistoryChart.destroy();
        }

        trainingHistoryChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    {
                        label: 'Training Loss',
                        data: data.loss,
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        borderWidth: 2,
                        tension: 0.1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Validation Loss',
                        data: data.val_loss,
                        borderColor: '#ffa94d',
                        backgroundColor: 'rgba(255, 168, 77, 0.1)',
                        borderWidth: 2,
                        tension: 0.1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Training Accuracy',
                        data: data.accuracy,
                        borderColor: '#51cf66',
                        backgroundColor: 'rgba(81, 207, 102, 0.1)',
                        borderWidth: 2,
                        tension: 0.1,
                        yAxisID: 'y1',
                        borderDash: [5, 5]
                    },
                    {
                        label: 'Validation Accuracy',
                        data: data.val_accuracy,
                        borderColor: '#4dabf7',
                        backgroundColor: 'rgba(77, 171, 247, 0.1)',
                        borderWidth: 2,
                        tension: 0.1,
                        yAxisID: 'y1',
                        borderDash: [5, 5]
                    }
                ]
            },
            options: {
                responsive: true,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Loss'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Accuracy'
                        },
                        grid: {
                            drawOnChartArea: false,
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading training history:', error);
    }
}

// Load ROC curve
async function loadROCCurve() {
    try {
        const response = await fetch('/api/graphs/roc-curve');
        const data = await response.json();
        
        if (data.error) {
            console.log('ROC curve data not available');
            return;
        }

        const ctx = document.getElementById('rocCurveChart');

        if (rocCurveChart) {
            rocCurveChart.destroy();
        }

        rocCurveChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: `ROC Curve (AUC = ${data.auc.toFixed(3)})`,
                        data: data.fpr.map((fpr, i) => ({
                            x: fpr,
                            y: data.tpr[i]
                        })),
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 2,
                        showLine: true,
                        tension: 0.1,
                        fill: true
                    },
                    {
                        label: 'Random Classifier',
                        data: [
                            {x: 0, y: 0},
                            {x: 1, y: 1}
                        ],
                        borderColor: '#ccc',
                        borderWidth: 1,
                        showLine: true,
                        borderDash: [5, 5],
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'False Positive Rate'
                        },
                        min: 0,
                        max: 1
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'True Positive Rate'
                        },
                        min: 0,
                        max: 1
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading ROC curve:', error);
    }
}

// Load Precision-Recall curve
async function loadPrecisionRecallCurve() {
    try {
        const response = await fetch('/api/graphs/precision-recall');
        const data = await response.json();
        
        if (data.error) {
            console.log('Precision-Recall curve data not available');
            return;
        }

        const ctx = document.getElementById('prCurveChart');

        if (prCurveChart) {
            prCurveChart.destroy();
        }

        prCurveChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: `Precision-Recall (AP = ${data.avg_precision.toFixed(3)})`,
                        data: data.recall.map((recall, i) => ({
                            x: recall,
                            y: data.precision[i]
                        })),
                        borderColor: '#f44336',
                        backgroundColor: 'rgba(244, 67, 54, 0.1)',
                        borderWidth: 2,
                        showLine: true,
                        tension: 0.1,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Recall'
                        },
                        min: 0,
                        max: 1
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Precision'
                        },
                        min: 0,
                        max: 1
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error loading Precision-Recall curve:', error);
    }
}
