/**
 * Generic Dashboard D3.js implementation
 * Strictly follows backend contract:
 * { "timestamp": "...", "metrics": { "value": number, "category": string }, "filters": { "region": string, "type": string } }
 */

const CONFIG = {
    API_URL: 'static/mock_api_data.json',
    CHART_SELECTOR: '#main-chart',
    TRANSITION_DURATION: 750
};

let chartState = {
    data: [],
    svg: null,
    width: 0,
    height: 0,
    margin: { top: 20, right: 30, bottom: 40, left: 60 }
};

const tooltip = d3.select("body").append("div").attr("class", "tooltip");

// --- API Simulation & State Management ---

async function fetchData() {
    showLoading(true);
    showError(false);
    
    try {
        // Simulating network delay for realism
        await new Promise(resolve => setTimeout(resolve, 800));
        
        const response = await fetch(CONFIG.API_URL);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const rawData = await response.json();
        
        // Strict contract adherence: we don't reshape the data.
        // We pass the raw array of contract objects directly to D3.
        chartState.data = rawData;
        
        updateUI();
        renderChart();
        showLoading(false);
        
    } catch (error) {
        console.error('API Fetch failed:', error);
        showLoading(false);
        showError(true);
    }
}

// --- UI Helpers ---

function showLoading(show) {
    document.getElementById('loading-overlay').classList.toggle('hidden', !show);
}

function showError(show) {
    document.getElementById('error-overlay').classList.toggle('hidden', !show);
    d3.select(CONFIG.CHART_SELECTOR).select('svg').style('opacity', show ? 0.3 : 1);
}

function updateUI() {
    if (chartState.data.length > 0) {
        // Extract filters from the first object as per contract (assuming homogeneous response for this demo)
        const filters = chartState.data[0].filters;
        document.getElementById('active-region').textContent = filters.region || 'All';
        document.getElementById('active-type').textContent = filters.type || 'All';
    }
}

// --- D3.js Rendering Logic ---

function initChart() {
    const container = d3.select(CONFIG.CHART_SELECTOR);
    const containerNode = container.node();
    
    // Set up responsive dimensions
    const width = containerNode.clientWidth;
    const height = containerNode.clientHeight;
    
    chartState.width = width - chartState.margin.left - chartState.margin.right;
    chartState.height = height - chartState.margin.top - chartState.margin.bottom;
    
    // Remove existing SVG if resizing
    container.select('svg').remove();
    
    chartState.svg = container.append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', `0 0 ${width} ${height}`)
        .append('g')
        .attr('transform', `translate(${chartState.margin.left},${chartState.margin.top})`);
        
    // Initialize axes groups
    chartState.svg.append("g").attr("class", "x-axis").attr("transform", `translate(0,${chartState.height})`);
    chartState.svg.append("g").attr("class", "y-axis");
}

function renderChart() {
    if (!chartState.svg) initChart();
    
    const svg = chartState.svg;
    const data = chartState.data;
    
    // Scales - Directly accessing nested contract properties
    const x = d3.scaleBand()
        .range([0, chartState.width])
        .domain(data.map(d => d.metrics.category))
        .padding(0.2);
        
    const y = d3.scaleLinear()
        .range([chartState.height, 0])
        .domain([0, d3.max(data, d => d.metrics.value) * 1.1]); // 10% headroom
        
    // Update axes with transitions
    svg.select(".x-axis")
        .transition().duration(CONFIG.TRANSITION_DURATION)
        .call(d3.axisBottom(x))
        .selectAll("text")
        .style("font-size", "14px");
        
    svg.select(".y-axis")
        .transition().duration(CONFIG.TRANSITION_DURATION)
        .call(d3.axisLeft(y).ticks(5))
        .selectAll("text")
        .style("font-size", "12px");
        
    // Data Binding Pattern (Enter, Update, Exit)
    // We key by category to ensure idempotent updates and proper object constancy
    const bars = svg.selectAll(".bar")
        .data(data, d => d.metrics.category);
        
    // 1. Exit: Remove old elements
    bars.exit()
        .transition().duration(CONFIG.TRANSITION_DURATION / 2)
        .attr("y", chartState.height)
        .attr("height", 0)
        .style("opacity", 0)
        .remove();
        
    // 2. Update: Transition existing elements to new positions
    bars.transition().duration(CONFIG.TRANSITION_DURATION)
        .attr("x", d => x(d.metrics.category))
        .attr("y", d => y(d.metrics.value))
        .attr("width", x.bandwidth())
        .attr("height", d => chartState.height - y(d.metrics.value));
        
    // 3. Enter: Add new elements
    const barsEnter = bars.enter()
        .append("rect")
        .attr("class", "bar")
        .attr("fill", "#0d6efd")
        .attr("rx", 4) // Rounded corners
        .attr("x", d => x(d.metrics.category))
        .attr("width", x.bandwidth())
        .attr("y", chartState.height) // Start at bottom for animation
        .attr("height", 0); // Start with 0 height
        
    // Enter transition
    barsEnter.transition().duration(CONFIG.TRANSITION_DURATION)
        .attr("y", d => y(d.metrics.value))
        .attr("height", d => chartState.height - y(d.metrics.value));
        
    // Interactivity on Enter + Update selection
    barsEnter.merge(bars)
        .on("mouseover", function(event, d) {
            d3.select(this)
                .transition().duration(200)
                .attr("fill", "#ffc107")
                .attr("opacity", 0.8);
                
            tooltip.style("opacity", 1)
                .html(`
                    <strong>${d.metrics.category}</strong><br/>
                    Value: ${d.metrics.value}<br/>
                    <small class="text-muted">${new Date(d.timestamp).toLocaleDateString()}</small>
                `)
                .style("left", (event.pageX + 15) + "px")
                .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", function() {
            d3.select(this)
                .transition().duration(200)
                .attr("fill", "#0d6efd")
                .attr("opacity", 1);
                
            tooltip.style("opacity", 0);
        });
}

// --- Event Listeners & Initialization ---

document.getElementById('refresh-btn').addEventListener('click', fetchData);
document.getElementById('retry-btn').addEventListener('click', fetchData);

// Handle window resize responsively
window.addEventListener('resize', () => {
    // Debounce resize
    clearTimeout(window.resizeTimer);
    window.resizeTimer = setTimeout(() => {
        initChart(); // Recompute dimensions
        if (chartState.data.length > 0) renderChart(); // Re-render with new scales
    }, 250);
});

// Kickoff
fetchData();
