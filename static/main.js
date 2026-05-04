// Constants
const MARGIN = { top: 20, right: 20, bottom: 20, left: 20 };
const EPOCHS = [0, 1, 3, 5, 10, 15];

// Custom tailored 10-color palette for digits (vibrant against dark mode)
const colorPalette = [
    "#f87171", // 0: Red
    "#fb923c", // 1: Orange
    "#facc15", // 2: Yellow
    "#4ade80", // 3: Green
    "#2dd4bf", // 4: Teal
    "#38bdf8", // 5: Light Blue
    "#818cf8", // 6: Indigo
    "#c084fc", // 7: Purple
    "#f472b6", // 8: Pink
    "#94a3b8"  // 9: Slate
];
const COLORS = d3.scaleOrdinal().domain(d3.range(10)).range(colorPalette);

// State
let appData = [];
let currentEpochIndex = 5; // default to max (Epoch 15)
let currentProjection = 'tsne';
let selectedMatrixCell = null; // for linking
let isDarkMode = false; // default light mode based on CSS
let currentFeatureMaps = null;
let currentPredictions = null;
let fmapPages = { conv2d_1: 0, conv2d_2: 0 };

// D3 Elements
let svgScatter, gScatter, xScale, yScale, circles, tooltip;
let svgMatrix, gMatrix;

/**
 * Initializes the dashboard application by fetching data and setting up visual components.
 */
async function init() {
    try {
        const response = await fetch('/data.json');
        if (!response.ok) throw new Error("Could not load data.json");
        appData = await response.json();
        
        setupTooltip();
        setupScatterPlot();
        setupMatrixSVG();
        setupControls();
        setupCanvas();
        
        updateScatterPlot(false); 
        updateConfusionMatrix(); 
    } catch (err) {
        console.error("Initialization error:", err);
    }
}

/**
 * Initializes the global D3 tooltip element.
 */
function setupTooltip() {
    tooltip = d3.select('body')
        .append('div')
        .attr('class', 'tooltip')
        .style('opacity', 0)
        .style('position', 'absolute')
        .style('background', 'var(--surface)')
        .style('border', '1px solid var(--glass-border)')
        .style('border-radius', '8px')
        .style('padding', '8px')
        .style('pointer-events', 'none')
        .style('z-index', 100)
        .style('box-shadow', '0 4px 12px rgba(0,0,0,0.5)');
}

/**
 * Initializes the SVG structure for the global scatter plot view.
 */
function setupScatterPlot() {
    const container = document.getElementById('scatter-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    svgScatter = d3.select('#scatter-container')
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', `0 0 ${width} ${height}`);

    const zoom = d3.zoom()
        .scaleExtent([0.5, 20])
        .on('zoom', (e) => {
            gScatter.attr('transform', e.transform);
        });

    svgScatter.call(zoom);

    gScatter = svgScatter.append('g');

    xScale = d3.scaleLinear().range([MARGIN.left, width - MARGIN.right]);
    yScale = d3.scaleLinear().range([height - MARGIN.bottom, MARGIN.top]);
    
    circles = gScatter.selectAll('circle')
        .data(appData, d => d.id)
        .enter()
        .append('circle')
        .attr('r', 4.5)
        .attr('fill', d => COLORS(d.true_label))
        .attr('opacity', 0.85)
        .attr('stroke', 'var(--point-stroke)')
        .attr('stroke-width', 1)
        .style('transition', 'opacity 0.3s, stroke 0.3s') 
        .on('mouseover', handleMouseOver)
        .on('mouseout', handleMouseOut);
}

/**
 * Updates the scatter plot coordinates and scales based on the active epoch and projection.
 * @param {boolean} transition - Whether to animate the update.
 */
function updateScatterPlot(transition = true) {
    const epochKey = String(EPOCHS[currentEpochIndex]);
    
    const xDomain = d3.extent(appData, d => d.epochs[epochKey][currentProjection][0]);
    const yDomain = d3.extent(appData, d => d.epochs[epochKey][currentProjection][1]);
    
    const xPadding = (xDomain[1] - xDomain[0]) * 0.1;
    const yPadding = (yDomain[1] - yDomain[0]) * 0.1;
    
    xScale.domain([xDomain[0] - xPadding, xDomain[1] + xPadding]);
    yScale.domain([yDomain[0] - yPadding, yDomain[1] + yPadding]);

    let selection = circles;
    if (transition) {
        selection = circles.transition().duration(1000).ease(d3.easeCubicInOut);
    }
    
    selection
        .attr('cx', d => xScale(d.epochs[epochKey][currentProjection][0]))
        .attr('cy', d => yScale(d.epochs[epochKey][currentProjection][1]));
}

function handleMouseOver(event, d) {
    d3.select(this)
        .attr('stroke', 'var(--text-primary)')
        .attr('stroke-width', 2)
        .attr('r', 8)
        .style('opacity', 1);
        
    tooltip.transition().duration(200).style('opacity', 1);
    
    const epochKey = String(EPOCHS[currentEpochIndex]);
    const currentPred = d.epochs[epochKey]['predicted_label'];
    const isCorrect = currentPred === d.true_label;
    const textColor = isCorrect ? 'var(--text-primary)' : '#f87171';
    
    tooltip.html(`
        <div style="display:flex; flex-direction:column; align-items:center; gap:8px;">
            <img src="${d.image_b64}" width="56" height="56" style="image-rendering:pixelated; border:1px solid var(--glass-border); border-radius:4px;" />
            <div style="font-size:13px; font-weight:500;">
                True: <span style="color:${COLORS(d.true_label)}">${d.true_label}</span> <br/>
                Pred: <span style="color:${textColor}">${currentPred}</span>
            </div>
        </div>
    `)
        .style('left', (event.pageX + 15) + 'px')
        .style('top', (event.pageY - 40) + 'px');
}

function handleMouseOut(event, d) {
    const epochKey = String(EPOCHS[currentEpochIndex]);
    const isFilteredOut = selectedMatrixCell && 
        !(d.true_label === selectedMatrixCell.t && d.epochs[epochKey].predicted_label === selectedMatrixCell.p);
    
    d3.select(this)
        .attr('stroke', 'var(--point-stroke)')
        .attr('stroke-width', 1)
        .attr('r', 4.5)
        .style('opacity', isFilteredOut ? 0.05 : 0.85);
        
    tooltip.transition().duration(300).style('opacity', 0);
}

/**
 * Initializes the SVG element for the diagnostic confusion matrix.
 */
function setupMatrixSVG() {
    const container = document.getElementById('matrix-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    svgMatrix = d3.select('#matrix-container')
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', `0 0 ${width} ${height}`);
        
    gMatrix = svgMatrix.append('g');
}

/**
 * Re-renders the confusion matrix based on the currently selected epoch's predictions.
 */
function updateConfusionMatrix() {
    const epochKey = String(EPOCHS[currentEpochIndex]);
    const container = document.getElementById('matrix-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    const margin = { top: 30, right: 20, bottom: 30, left: 30 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const gridSize = Math.min(innerWidth, innerHeight) / 10;
    
    gMatrix.selectAll('*').remove();
    gMatrix.attr('transform', `translate(${(width - gridSize*10)/2 + 10}, ${(height - gridSize*10)/2 + 10})`);
    
    const matrix = Array(10).fill(0).map(() => Array(10).fill(0));
    appData.forEach(d => {
        matrix[d.true_label][d.epochs[epochKey].predicted_label]++;
    });
    
    const maxVal = d3.max(matrix.flat());
    const colorScale = d3.scaleSequential(d3.interpolateBlues).domain([0, maxVal]);
    const errorColorScale = d3.scaleSequential(d3.interpolateReds).domain([0, maxVal * 0.1]); 
    
    const labels = d3.range(10);
    gMatrix.selectAll(".rowLabel")
        .data(labels)
        .enter().append("text")
        .text(d => d)
        .attr("x", -10)
        .attr("y", (d, i) => i * gridSize + gridSize / 2)
        .style("text-anchor", "end")
        .style("alignment-baseline", "middle")
        .style("font-size", "12px")
        .style("fill", "var(--text-secondary)");
        
    gMatrix.selectAll(".colLabel")
        .data(labels)
        .enter().append("text")
        .text(d => d)
        .attr("x", (d, i) => i * gridSize + gridSize / 2)
        .attr("y", -10)
        .style("text-anchor", "middle")
        .style("font-size", "12px")
        .style("fill", "var(--text-secondary)");
        
    gMatrix.append("text").text("Predicted").attr("x", gridSize*5).attr("y", -25).style("text-anchor", "middle").style("font-size", "12px").style("fill", "var(--text-primary)");
    gMatrix.append("text").text("True").attr("transform", "rotate(-90)").attr("x", -gridSize*5).attr("y", -25).style("text-anchor", "middle").style("font-size", "12px").style("fill", "var(--text-primary)");

    const strokeColor = getComputedStyle(document.documentElement).getPropertyValue('--matrix-stroke').trim() || '#000';

    for (let i = 0; i < 10; i++) {
        for (let j = 0; j < 10; j++) {
            const val = matrix[i][j];
            const isDiagonal = i === j;
            const fill = isDiagonal ? colorScale(val) : errorColorScale(val);
            const isSelected = selectedMatrixCell && selectedMatrixCell.t === i && selectedMatrixCell.p === j;
            
            const cellGroup = gMatrix.append('g')
                .style("cursor", "pointer")
                .on("mouseover", function() { 
                    if(!isSelected) d3.select(this).select('rect').attr("stroke", strokeColor).attr("stroke-width", 2); 
                })
                .on("mouseout", function() { 
                    if(!isSelected) d3.select(this).select('rect').attr("stroke", isSelected ? "#facc15" : "none"); 
                })
                .on("click", () => filterScatterPlot(i, j));

            cellGroup.append("rect")
                .attr("x", j * gridSize)
                .attr("y", i * gridSize)
                .attr("width", gridSize - 2)
                .attr("height", gridSize - 2)
                .attr("fill", val === 0 ? "var(--glass-border)" : fill)
                .attr("rx", 2)
                .attr("stroke", isSelected ? "#facc15" : "none")
                .attr("stroke-width", isSelected ? 3 : 0);
                
            if (val > 0) {
                cellGroup.append("text")
                    .attr("x", j * gridSize + gridSize / 2)
                    .attr("y", i * gridSize + gridSize / 2)
                    .text(val)
                    .style("text-anchor", "middle")
                    .style("alignment-baseline", "middle")
                    .style("font-size", "10px")
                    .style("pointer-events", "none")
                    .style("fill", val > maxVal * 0.5 ? "white" : "var(--text-primary)");
            }
        }
    }
}

/**
 * Toggles the scatter plot filtering based on a selected confusion matrix cell.
 * @param {number} t - The true label class index.
 * @param {number} p - The predicted label class index.
 */
function filterScatterPlot(t, p) {
    if (selectedMatrixCell && selectedMatrixCell.t === t && selectedMatrixCell.p === p) {
        clearFilter();
    } else {
        selectedMatrixCell = {t, p};
        document.getElementById('reset-filter-btn').style.display = 'block';
        applyScatterFilter();
        updateConfusionMatrix(); 
    }
}

/**
 * Clears the active matrix filter and restores full visibility to the scatter plot.
 */
function clearFilter() {
    selectedMatrixCell = null;
    document.getElementById('reset-filter-btn').style.display = 'none';
    circles.style('opacity', 0.85).style('pointer-events', 'all');
    updateConfusionMatrix();
}

/**
 * Applies the opacity filter to the scatter plot nodes based on the selected matrix cell.
 */
function applyScatterFilter() {
    if (!selectedMatrixCell) return;
    const epochKey = String(EPOCHS[currentEpochIndex]);
    const t = selectedMatrixCell.t;
    const p = selectedMatrixCell.p;
    
    circles.style('opacity', d => {
        return (d.true_label === t && d.epochs[epochKey].predicted_label === p) ? 1.0 : 0.05;
    })
    .style('pointer-events', d => {
        return (d.true_label === t && d.epochs[epochKey].predicted_label === p) ? 'all' : 'none';
    });
    
    circles.filter(d => d.true_label === t && d.epochs[epochKey].predicted_label === p).raise();
}

// === LOCAL VIEW (CANVAS & INFERENCE) ===
let isDrawing = false;
let ctx;

function setupCanvas() {
    const canvas = document.getElementById('drawing-canvas');
    ctx = canvas.getContext('2d');
    
    // Fill black
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    const getPos = (e) => {
        const rect = canvas.getBoundingClientRect();
        return {
            x: (e.clientX || e.touches[0].clientX) - rect.left,
            y: (e.clientY || e.touches[0].clientY) - rect.top
        };
    };

    const startDraw = (e) => {
        isDrawing = true;
        const pos = getPos(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        e.preventDefault();
    };

    const draw = (e) => {
        if (!isDrawing) return;
        const pos = getPos(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        e.preventDefault();
    };

    const stopDraw = () => {
        if (isDrawing) {
            ctx.closePath();
            isDrawing = false;
        }
    };

    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDraw);
    canvas.addEventListener('mouseout', stopDraw);
    
    // Touch support
    canvas.addEventListener('touchstart', startDraw);
    canvas.addEventListener('touchmove', draw);
    canvas.addEventListener('touchend', stopDraw);
    
    document.getElementById('clear-btn').addEventListener('click', () => {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        document.getElementById('bar-chart-container').innerHTML = '';
        document.getElementById('feature-maps-container').innerHTML = '';
    });
    
    document.getElementById('submit-btn').addEventListener('click', performInference);
}

/**
 * Transmits the canvas content to the backend for inference and processes the response.
 */
async function performInference() {
    const canvas = document.getElementById('drawing-canvas');
    const dataURL = canvas.toDataURL('image/png');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataURL })
        });
        
        if (!response.ok) throw new Error("Inference failed");
        
        const result = await response.json();
        console.log('Layer 1 maps:', result.feature_maps.conv2d_1.length, '| Layer 2 maps:', result.feature_maps.conv2d_2.length);
        
        currentFeatureMaps = result.feature_maps;
        currentPredictions = result.predictions;
        fmapPages = { conv2d_1: 0, conv2d_2: 0 };
        
        renderBarChart(result.predictions);
        renderFeatureMaps();
    } catch (err) {
        console.error("Error during inference:", err);
    }
}

/**
 * Renders the prediction confidence distribution as a D3 bar chart.
 * @param {number[]} predictions - An array of class probabilities.
 */
function renderBarChart(predictions) {
    const container = document.getElementById('bar-chart-container');
    container.innerHTML = '';
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    const svgBar = d3.select(container).append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', `0 0 ${width} ${height}`);
        
    const margin = {top: 10, right: 10, bottom: 25, left: 30};
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    
    const g = svgBar.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    
    const x = d3.scaleBand().range([0, innerW]).domain(d3.range(10)).padding(0.2);
    const y = d3.scaleLinear().range([innerH, 0]).domain([0, 1]);
    
    g.append('g')
       .attr('transform', `translate(0, ${innerH})`)
       .call(d3.axisBottom(x).tickSizeOuter(0))
       .style("font-family", "Outfit")
       .style("color", "var(--text-secondary)");
       
    g.append('g')
       .call(d3.axisLeft(y).ticks(4).tickFormat(d3.format(".0%")))
       .style("font-family", "Outfit")
       .style("color", "var(--text-secondary)");
       
    g.selectAll('rect')
       .data(predictions)
       .enter()
       .append('rect')
       .attr('x', (d, i) => x(i))
       .attr('y', d => y(d))
       .attr('width', x.bandwidth())
       .attr('height', d => innerH - y(d))
       .attr('fill', (d, i) => COLORS(i))
       .attr('rx', 2);
}

function renderFeatureMaps() {
    if (!currentFeatureMaps) return;
    const container = document.getElementById('feature-maps-container');
    container.innerHTML = '';
    
    const createSection = (title, maps, layerKey) => {
        const div = document.createElement('div');
        div.style.marginBottom = '12px';
        div.style.width = '100%';
        
        const ITEMS_PER_PAGE = window.innerWidth <= 1200 ? 8 : 16;
        const totalPages = Math.ceil(maps.length / ITEMS_PER_PAGE);
        const currentPage = fmapPages[layerKey];
        const startIdx = currentPage * ITEMS_PER_PAGE;
        const endIdx = Math.min(startIdx + ITEMS_PER_PAGE, maps.length);
        
        const header = document.createElement('div');
        header.style.display = 'flex';
        header.style.justifyContent = 'space-between';
        header.style.alignItems = 'center';
        header.style.borderBottom = '1px solid var(--panel-inset-border)';
        header.style.paddingBottom = '4px';
        header.style.marginBottom = '8px';
        
        const btnStyle = 'cursor:pointer; background:none; border:none; color:var(--accent); font-weight:600; padding: 2px 6px; font-family: inherit; font-size: 0.8rem;';
        const prevDisabled = currentPage === 0 ? 'disabled style="opacity:0.4; cursor:default;"' : '';
        const nextDisabled = currentPage >= totalPages - 1 ? 'disabled style="opacity:0.4; cursor:default;"' : '';
        
        header.innerHTML = `
            <h4 style="font-size:0.8rem; color:var(--text-secondary); letter-spacing:0.5px; margin:0;">${title}</h4>
            <div style="font-size:0.75rem; color:var(--text-secondary); display:flex; gap:6px; align-items:center;">
                <button class="page-btn" data-dir="-1" style="${btnStyle}" ${prevDisabled}>&lt; Prev</button>
                <span>Showing ${startIdx + 1}-${endIdx} of ${maps.length}</span>
                <button class="page-btn" data-dir="1" style="${btnStyle}" ${nextDisabled}>Next &gt;</button>
            </div>
        `;
        
        const btns = header.querySelectorAll('.page-btn');
        btns.forEach(btn => {
            if (!btn.disabled) {
                btn.addEventListener('click', (e) => {
                    const dir = parseInt(e.target.dataset.dir);
                    fmapPages[layerKey] += dir;
                    renderFeatureMaps();
                });
            }
        });
        
        div.appendChild(header);
        
        const grid = document.createElement('div');
        grid.style.display = 'grid';
        grid.style.gridTemplateColumns = 'repeat(auto-fill, 80px)';
        grid.style.gap = '8px';
        grid.style.width = '100%';
        
        const currentMaps = maps.slice(startIdx, endIdx);
        currentMaps.forEach(b64 => {
            const img = document.createElement('img');
            img.src = b64;
            img.className = 'fmap-img';
            img.style.width = '80px';
            img.style.height = '80px';
            grid.appendChild(img);
        });
        
        div.appendChild(grid);
        container.appendChild(div);
    };
    
    createSection('Conv2D Layer 1', currentFeatureMaps.conv2d_1, 'conv2d_1');
    createSection('Conv2D Layer 2', currentFeatureMaps.conv2d_2, 'conv2d_2');
}

// === CONTROLS ===
function setupControls() {
    const epochSlider = document.getElementById('epoch-slider');
    const epochDisplay = document.getElementById('epoch-display');
    const projectionSelect = document.getElementById('projection-select');
    const themeToggleBtn = document.getElementById('theme-toggle');
    const resetFilterBtn = document.getElementById('reset-filter-btn');

    epochSlider.addEventListener('input', (e) => {
        currentEpochIndex = parseInt(e.target.value);
        epochDisplay.textContent = EPOCHS[currentEpochIndex];
        updateScatterPlot(true);
        updateConfusionMatrix(); 
        if (selectedMatrixCell) applyScatterFilter(); 
    });

    projectionSelect.addEventListener('change', (e) => {
        currentProjection = e.target.value;
        updateScatterPlot(true);
    });

    themeToggleBtn.addEventListener('click', () => {
        isDarkMode = !isDarkMode;
        document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
        themeToggleBtn.textContent = isDarkMode ? '☀️' : '🌙';
        
        updateConfusionMatrix();
    });

    resetFilterBtn.addEventListener('click', clearFilter);
}

window.addEventListener('resize', () => {
    const scatterContainer = document.getElementById('scatter-container');
    if (svgScatter) {
        svgScatter.attr('viewBox', `0 0 ${scatterContainer.clientWidth} ${scatterContainer.clientHeight}`);
        xScale.range([MARGIN.left, scatterContainer.clientWidth - MARGIN.right]);
        yScale.range([scatterContainer.clientHeight - MARGIN.bottom, MARGIN.top]);
        updateScatterPlot(false);
    }
    
    const matrixContainer = document.getElementById('matrix-container');
    if (svgMatrix) {
        svgMatrix.attr('viewBox', `0 0 ${matrixContainer.clientWidth} ${matrixContainer.clientHeight}`);
        updateConfusionMatrix();
    }
    
    if (currentPredictions) {
        renderBarChart(currentPredictions);
    }
    
    if (currentFeatureMaps) {
        renderFeatureMaps();
    }
});

document.addEventListener('DOMContentLoaded', init);
