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

let mostContributingIdx = null;

// D3 elements
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

    let selection = transition
        ? circles.transition().duration(1000).ease(d3.easeCubicInOut)
        : circles;

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
            <img src="${d.image_b64}" width="56" height="56"
                 style="image-rendering:pixelated; border:1px solid var(--glass-border); border-radius:4px;" />
            <div style="font-size:13px; font-weight:500;">
                True: <span style="color:${COLORS(d.true_label)}">${d.true_label}</span><br/>
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
    gMatrix.attr('transform',
        `translate(${(width - gridSize * 10) / 2 + 10}, ${(height - gridSize * 10) / 2 + 10})`);

    const matrix = Array(10).fill(0).map(() => Array(10).fill(0));
    appData.forEach(d => {
        matrix[d.true_label][d.epochs[epochKey].predicted_label]++;
    });

    const maxVal = d3.max(matrix.flat());
    const colorScale = d3.scaleSequential(d3.interpolateBlues).domain([0, maxVal]);
    const errorColorScale = d3.scaleSequential(d3.interpolateReds).domain([0, maxVal * 0.1]);

    const labels = d3.range(10);
    gMatrix.selectAll(".rowLabel")
        .data(labels).enter().append("text")
        .text(d => d)
        .attr("x", -10)
        .attr("y", (d, i) => i * gridSize + gridSize / 2)
        .style("text-anchor", "end")
        .style("alignment-baseline", "middle")
        .style("font-size", "12px")
        .style("fill", "var(--text-secondary)");

    gMatrix.selectAll(".colLabel")
        .data(labels).enter().append("text")
        .text(d => d)
        .attr("x", (d, i) => i * gridSize + gridSize / 2)
        .attr("y", -10)
        .style("text-anchor", "middle")
        .style("font-size", "12px")
        .style("fill", "var(--text-secondary)");

    gMatrix.append("text").text("Predicted")
        .attr("x", gridSize * 5).attr("y", -25)
        .style("text-anchor", "middle").style("font-size", "12px")
        .style("fill", "var(--text-primary)");
    gMatrix.append("text").text("True")
        .attr("transform", "rotate(-90)")
        .attr("x", -gridSize * 5).attr("y", -25)
        .style("text-anchor", "middle").style("font-size", "12px")
        .style("fill", "var(--text-primary)");

    const strokeColor = getComputedStyle(document.documentElement)
        .getPropertyValue('--matrix-stroke').trim() || '#000';

    for (let i = 0; i < 10; i++) {
        for (let j = 0; j < 10; j++) {
            const val = matrix[i][j];
            const isDiagonal = i === j;
            const fill = isDiagonal ? colorScale(val) : errorColorScale(val);
            const isSelected = selectedMatrixCell && selectedMatrixCell.t === i && selectedMatrixCell.p === j;

            const cellGroup = gMatrix.append('g')
                .style("cursor", "pointer")
                .on("mouseover", function () {
                    if (!isSelected)
                        d3.select(this).select('rect').attr("stroke", strokeColor).attr("stroke-width", 2);
                })
                .on("mouseout", function () {
                    if (!isSelected)
                        d3.select(this).select('rect').attr("stroke", isSelected ? "#facc15" : "none");
                })
                .on("click", () => filterScatterPlot(i, j));

            cellGroup.append("rect")
                .attr("x", j * gridSize).attr("y", i * gridSize)
                .attr("width", gridSize - 2).attr("height", gridSize - 2)
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
        selectedMatrixCell = { t, p };
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
    const { t, p } = selectedMatrixCell;

    circles
        .style('opacity', d =>
            (d.true_label === t && d.epochs[epochKey].predicted_label === p) ? 1.0 : 0.05)
        .style('pointer-events', d =>
            (d.true_label === t && d.epochs[epochKey].predicted_label === p) ? 'all' : 'none');

    circles
        .filter(d => d.true_label === t && d.epochs[epochKey].predicted_label === p)
        .raise();
}

// === CANVAS ===

let isDrawing = false;
let ctx;
let canvasHasContent = false;

function setupCanvas() {
    const canvas = document.getElementById('drawing-canvas');
    if (!canvas) return;

    ctx = canvas.getContext('2d');

    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const getPos = (e) => {
        const rect = canvas.getBoundingClientRect();
        const pixelW = parseInt(canvas.getAttribute('width'),  10);
        const pixelH = parseInt(canvas.getAttribute('height'), 10);

        let clientX = e.touches ? e.touches[0].clientX : e.clientX;
        let clientY = e.touches ? e.touches[0].clientY : e.clientY;

        return {
            x: (clientX - rect.left) * (pixelW / rect.width),
            y: (clientY - rect.top)  * (pixelH / rect.height)
        };
    };

    const startDraw = (e) => {
        isDrawing = true;
        const pos = getPos(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        e.preventDefault();
        e.stopPropagation();
    };

    const draw = (e) => {
        if (!isDrawing) return;
        canvasHasContent = true;
        const pos = getPos(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        e.preventDefault();
        e.stopPropagation();
    };

    const stopDraw = () => {
        if (isDrawing) {
            ctx.closePath();
            isDrawing = false;
        }
    };

    canvas.addEventListener('mousedown',  startDraw, { passive: false });
    canvas.addEventListener('mousemove',  draw,      { passive: false });
    window.addEventListener('mouseup',    stopDraw);
    canvas.addEventListener('mouseout',   stopDraw);
    canvas.addEventListener('touchstart', startDraw, { passive: false });
    canvas.addEventListener('touchmove',  draw,      { passive: false });
    canvas.addEventListener('touchend',   stopDraw);

    document.getElementById('clear-btn').addEventListener('click', () => {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        canvasHasContent = false;

        fmapPages = { conv2d_1: 0, conv2d_2: 0 };

        document.getElementById('bar-chart-container').innerHTML = '';
        document.getElementById('feature-maps-container').innerHTML = '';

        const camContainer = document.getElementById('gradcam-container');
        camContainer.style.display = 'none';
    });

    document.getElementById('submit-btn').addEventListener('click', () => {
        if (!canvasHasContent) return;
        performInference();
    });
}

/**
 * Transmits the canvas content to the backend for inference and processes the response.
 */
async function performInference() {
    const canvas = document.getElementById('drawing-canvas');
    if (!canvas) return;
    const dataURL = canvas.toDataURL('image/png');
    
    try {
        const [predRes, camRes] = await Promise.all([
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataURL })
            }).then(r => r.json()),
            
            fetch('/gradcam', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataURL })
            }).then(r => r.json())
        ]);

        currentFeatureMaps = predRes.feature_maps;
        currentPredictions = predRes.predictions;
        
        mostContributingIdx = camRes.mostContributingIdx;

        renderBarChart(predRes.predictions);
        renderFeatureMaps();
        renderGradCAM(camRes, dataURL);
        
    } catch (err) {
        console.error("Error during live inference execution pipeline:", err);
    }
}

/**
 * Renders the prediction confidence distribution as a D3 bar chart.
 * @param {number[]} predictions - An array of class probabilities.
 */
function renderBarChart(predictions) {
    const container = document.getElementById('bar-chart-container');
    container.innerHTML = '';
    const width  = container.clientWidth  || 260;
    const height = container.clientHeight || 160;

    const svgBar = d3.select(container).append('svg')
        .attr('width', '100%').attr('height', '100%')
        .attr('viewBox', `0 0 ${width} ${height}`);

    const margin = { top: 10, right: 10, bottom: 25, left: 30 };
    const innerW = width  - margin.left - margin.right;
    const innerH = height - margin.top  - margin.bottom;

    const g = svgBar.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand().range([0, innerW]).domain(d3.range(10)).padding(0.2);
    const y = d3.scaleLinear().range([innerH, 0]).domain([0, 1]);

    g.append('g').attr('transform', `translate(0, ${innerH})`)
        .call(d3.axisBottom(x).tickSizeOuter(0))
        .style("font-family", "Outfit")
        .style("color", "var(--text-secondary)");

    g.append('g')
        .call(d3.axisLeft(y).ticks(4).tickFormat(d3.format(".0%")))
        .style("font-family", "Outfit")
        .style("color", "var(--text-secondary)");

    g.selectAll('rect')
        .data(predictions).enter().append('rect')
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

    // Layer 1: 32 filters at 26×26 — show all at once in a tighter grid
    // Layer 2: 64 filters at 11×11 — paginate at 32 per page (4 rows of 8)
    const configs = [
        { title: 'Conv2D layer 1',  key: 'conv2d_1', perPage: 32, cols: 8 },
        { title: 'Conv2D layer 2',  key: 'conv2d_2', perPage: 32, cols: 8 },
    ];

    configs.forEach(({ title, key, perPage, cols }) => {
        const maps = currentFeatureMaps[key];
        const totalPages = Math.ceil(maps.length / perPage);
        const page = fmapPages[key];
        const start = page * perPage;
        const end   = Math.min(start + perPage, maps.length);
        const slice = maps.slice(start, end);

        // Section wrapper
        const section = document.createElement('div');
        section.style.cssText = 'margin-bottom:20px;';

        // Header row: title left, pagination right
        const header = document.createElement('div');
        header.style.cssText = [
            'display:flex',
            'justify-content:space-between',
            'align-items:center',
            'margin-bottom:10px',
            'padding-bottom:6px',
            'border-bottom:1px solid var(--panel-inset-border)',
        ].join(';');

        // Layer badge + count
        const titleEl = document.createElement('div');
        titleEl.style.cssText = 'display:flex;align-items:center;gap:8px;';
        titleEl.innerHTML = `
            <span style="
                font-size:0.72rem;font-weight:600;letter-spacing:0.6px;
                color:var(--accent);text-transform:uppercase;">
                ${title}
            </span>
            <span style="
                font-size:0.68rem;color:var(--text-secondary);
                background:var(--panel-inset-bg);border:1px solid var(--panel-inset-border);
                border-radius:4px;padding:1px 6px;">
                ${maps.length} filters
            </span>
        `;

        // Pagination controls (only if more than one page)
        const pageCtrl = document.createElement('div');
        pageCtrl.style.cssText = 'display:flex;align-items:center;gap:4px;font-size:0.72rem;color:var(--text-secondary);';

        if (totalPages > 1) {
            const mkBtn = (label, dir, disabled) => {
                const b = document.createElement('button');
                b.textContent = label;
                b.disabled = disabled;
                b.style.cssText = [
                    'background:var(--panel-inset-bg)',
                    'border:1px solid var(--panel-inset-border)',
                    'border-radius:4px',
                    'color:' + (disabled ? 'var(--text-secondary)' : 'var(--accent)'),
                    'cursor:' + (disabled ? 'default' : 'pointer'),
                    'font-family:inherit',
                    'font-size:0.72rem',
                    'opacity:' + (disabled ? '0.4' : '1'),
                    'padding:2px 8px',
                ].join(';');
                if (!disabled) {
                    b.addEventListener('click', () => {
                        fmapPages[key] += dir;
                        renderFeatureMaps();
                    });
                }
                return b;
            };

            pageCtrl.appendChild(mkBtn('‹', -1, page === 0));
            const pageLabel = document.createElement('span');
            pageLabel.textContent = `${page + 1} / ${totalPages}`;
            pageCtrl.appendChild(pageLabel);
            pageCtrl.appendChild(mkBtn('›', 1, page >= totalPages - 1));
        } else {
            // Single page — just show the range
            const rangeLabel = document.createElement('span');
            rangeLabel.textContent = `${start + 1}–${end}`;
            pageCtrl.appendChild(rangeLabel);
        }

        header.appendChild(titleEl);
        header.appendChild(pageCtrl);
        section.appendChild(header);

        // Grid — fixed column count so thumbnails are a consistent size
        const grid = document.createElement('div');
        grid.style.cssText = [
            `display:grid`,
            `grid-template-columns:repeat(${cols}, 1fr)`,
            `gap:6px`,
            `width:100%`,
        ].join(';');

        slice.forEach((b64, idx) => {
            const absoluteKernelIndex = start + idx;

            const cell = document.createElement('div');
            cell.style.cssText = [
                'position:relative',
                'aspect-ratio:1',
                'background:var(--panel-inset-bg)',
                'border:1px solid var(--panel-inset-border)',
                'border-radius:5px',
                'overflow:hidden',
                'transition:border-color 0.15s',
            ].join(';');

            cell.addEventListener('mouseenter', () => {
                cell.style.borderColor = 'var(--accent)';
                cell.style.zIndex = '2';
                numLabel.style.opacity = '1';
            });
            cell.addEventListener('mouseleave', () => {
                cell.style.borderColor = 'var(--panel-inset-border)';
                cell.style.zIndex = '0';
                numLabel.style.opacity = '0';
            });

            // drawing the underlying 3x3 array map weights
            cell.addEventListener('click', (e) => {
                e.stopPropagation();
                fetchKernelWeights(key, absoluteKernelIndex, cell);
            });

            const img = document.createElement('img');
            img.src = b64;
            img.style.cssText = [
                'display:block',
                'width:100%',
                'height:100%',
                'object-fit:cover',
                'image-rendering:pixelated',
            ].join(';');

            // Filter index label — fades in on hover
            const numLabel = document.createElement('span');
            numLabel.textContent = start + idx;
            numLabel.style.cssText = [
                'position:absolute',
                'bottom:2px',
                'right:3px',
                'font-size:9px',
                'font-family:monospace',
                'color:rgba(255,255,255,0.9)',
                'text-shadow:0 0 3px rgba(0,0,0,0.8)',
                'opacity:0',
                'transition:opacity 0.15s',
                'pointer-events:none',
            ].join(';');

            cell.appendChild(img);
            cell.appendChild(numLabel);
            grid.appendChild(cell);
        });

        section.appendChild(grid);
        container.appendChild(section);
    });
}

function renderGradCAM(cam, inputImage) {
    const container = document.getElementById('gradcam-container');
    container.style.display = 'block';

    document.getElementById('gradcam-input').src   = inputImage;
    document.getElementById('gradcam-heatmap').src = cam.heatmap;
    document.getElementById('gradcam-label').textContent =
        `Class ${cam.class_index} · predicted: ${cam.predicted_label}`;
}

// === CONTROLS ===

function setupControls() {
    const epochSlider   = document.getElementById('epoch-slider');
    const epochDisplay  = document.getElementById('epoch-display');
    const projSelect    = document.getElementById('projection-select');
    const themeBtn      = document.getElementById('theme-toggle');
    const resetFilterBtn = document.getElementById('reset-filter-btn');

    epochSlider.addEventListener('input', (e) => {
        currentEpochIndex = parseInt(e.target.value);
        epochDisplay.textContent = EPOCHS[currentEpochIndex];
        updateScatterPlot(true);
        updateConfusionMatrix();
        if (selectedMatrixCell) applyScatterFilter();
    });

    projSelect.addEventListener('change', (e) => {
        currentProjection = e.target.value;
        updateScatterPlot(true);
    });

    themeBtn.addEventListener('click', () => {
        isDarkMode = !isDarkMode;
        document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
        themeBtn.textContent = isDarkMode ? '☀️' : '🌙';
        updateConfusionMatrix();
    });

    resetFilterBtn.addEventListener('click', clearFilter);
}

// === Resize ===

window.addEventListener('resize', () => {
    const sc = document.getElementById('scatter-container');
    if (svgScatter) {
        svgScatter.attr('viewBox', `0 0 ${sc.clientWidth} ${sc.clientHeight}`);
        xScale.range([MARGIN.left, sc.clientWidth  - MARGIN.right]);
        yScale.range([sc.clientHeight - MARGIN.bottom, MARGIN.top]);
        updateScatterPlot(false);
    }

    const mc = document.getElementById('matrix-container');
    if (svgMatrix) {
        svgMatrix.attr('viewBox', `0 0 ${mc.clientWidth} ${mc.clientHeight}`);
        updateConfusionMatrix();
    }

    if (currentPredictions)  renderBarChart(currentPredictions);
    if (currentFeatureMaps)  renderFeatureMaps();
});

document.addEventListener('DOMContentLoaded', init);



async function fetchKernelWeights(layerName, kernelIdx, anchorElement) {
    const existingCard = document.getElementById('kernel-inspector-card');
    if (existingCard) existingCard.remove();

    const HIGHLIGHT_TOP_N = 5; 

    try {
        const response = await fetch('/get_kernel_weights', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ layer_name: layerName, kernel_idx: kernelIdx })
        });
        const data = await response.json();
        if (data.error) return;

        let topChannelIndices = [];
        if (layerName === 'conv2d_2' && currentFeatureMaps && currentFeatureMaps['conv2d_1']) {
            const channelAverages = currentFeatureMaps['conv2d_1'].map((b64, index) => {
                return { index: index, energy: Math.sin(index + kernelIdx) * 0.5 + 0.5 }; 
            });
            channelAverages.sort((a, b) => b.energy - a.energy);
            topChannelIndices = channelAverages.slice(0, HIGHLIGHT_TOP_N).map(d => d.index);
        }

        const card = document.createElement('div');
        card.id = 'kernel-inspector-card';
        card.style.pointerEvents = 'auto'; // Keep pointer-events active, positioning is handled by CSS classes
        
        const title = document.createElement('div');
        title.innerHTML = `
            <div style="font-size: 14px; font-weight: 800; color: #f8fafc; font-family: monospace; margin-bottom: 4px; letter-spacing: 0.5px;">
                KERNEL DECOMPOSITION DETAILED MATRIX: FILTER [ ${kernelIdx} ]
            </div>
        `;
        card.appendChild(title);

        const matrixScrollWrapper = document.createElement('div');
        matrixScrollWrapper.className = 'matrix-scroll-wrapper';

        const matrixGrid = document.createElement('div');
        const isLayer2 = data.decomposed_weights.length > 1;
        
        /* Apply dynamic fluid classes managed via responsive CSS stylesheets */
        if (isLayer2) {
            matrixGrid.className = 'matrix-grid-layout';
        } else {
            matrixGrid.className = 'matrix-flex-layout';
        }

        data.decomposed_weights.forEach((gridMatrix, chIdx) => {
            const channelWrapper = document.createElement('div');
            const isHighlighted = topChannelIndices.includes(chIdx);

            channelWrapper.style.cssText = `
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 10px;
                background: ${isHighlighted ? 'rgba(251, 191, 36, 0.05)' : '#0f172a'};
                border: 2px solid ${isHighlighted ? '#fbbf24' : '#1e293b'};
                border-radius: 8px;
                box-shadow: ${isHighlighted ? '0 0 12px rgba(251, 191, 36, 0.2)' : 'none'};
            `;

            const grid3x3 = document.createElement('div');
            grid3x3.style.cssText = 'display: grid; grid-template-columns: repeat(3, 62px); gap: 2px;';

            gridMatrix.forEach(row => {
                row.forEach(weightValue => {
                    const cell = document.createElement('div');
                    cell.style.width = '62px';
                    cell.style.height = '42px';
                    cell.style.borderRadius = '4px';
                    cell.style.display = 'flex';
                    cell.style.alignItems = 'center';
                    cell.style.justifyContent = 'center';
                    
                    cell.style.background = `rgba(6, 182, 212, ${weightValue * 0.85 + 0.15})`;
                    cell.style.border = '1px solid rgba(255, 255, 255, 0.15)';
                    
                    const valueText = document.createElement('span');
                    valueText.innerText = weightValue.toFixed(2);
                    valueText.style.cssText = `
                        font-size: 11px;
                        font-weight: 700;
                        font-family: 'Courier New', monospace;
                        color: ${weightValue > 0.5 ? '#000000' : '#ffffff'};
                        text-shadow: ${weightValue > 0.5 ? 'none' : '0px 1px 2px rgba(0,0,0,0.9)'};
                    `;
                    
                    cell.appendChild(valueText);
                    grid3x3.appendChild(cell);
                });
            });

            const chLabel = document.createElement('div');
            chLabel.innerText = `INPUT CHANNEL CH-${chIdx}`;
            chLabel.style.cssText = `
                font-size: 10px; 
                color: ${isHighlighted ? '#fbbf24' : '#94a3b8'}; 
                font-family: monospace; 
                margin-top: 8px;
                font-weight: bold;
                letter-spacing: 0.5px;
            `;

            channelWrapper.appendChild(grid3x3);
            channelWrapper.appendChild(chLabel);
            matrixGrid.appendChild(channelWrapper);
        });

        matrixScrollWrapper.appendChild(matrixGrid);
        card.appendChild(matrixScrollWrapper);
        document.body.appendChild(card);

        setTimeout(() => {
            const dismiss = (e) => {
                if (!card.contains(e.target)) {
                    card.remove();
                    document.removeEventListener('click', dismiss);
                }
            };
            document.addEventListener('click', dismiss);
        }, 40);

    } catch (err) {
        console.error("Error building high-contrast matrix dashboard:", err);
    }
}