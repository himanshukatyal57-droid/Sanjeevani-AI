
// ===================================
// SANJEEVANI AI - ROUTE MAP
// Draws the A* search as SVG from our own graph data, with no map API.
// ===================================

const SVG_NS = "http://www.w3.org/2000/svg";
const MAP_WIDTH = 1000;
const MAP_PADDING = 40;
const HEAVY_TRAFFIC_CONGESTION = 1.7;
const EXPLORE_STEP_MS = 280;
const ROUTE_STEP_MS = 320;

function svgElement(tag, attributes, parent) {
    const element = document.createElementNS(SVG_NS, tag);

    Object.entries(attributes).forEach(([name, value]) => {
        element.setAttribute(name, value);
    });

    if (parent) {
        parent.appendChild(element);
    }

    return element;
}

// Fits latitude and longitude to the drawing, keeping real proportions at this latitude
function createProjection(locations) {
    const latitudes = locations.map(location => location.lat);
    const longitudes = locations.map(location => location.lon);
    const minLon = Math.min(...longitudes);
    const maxLon = Math.max(...longitudes);
    const minLat = Math.min(...latitudes);
    const maxLat = Math.max(...latitudes);
    const lonScale = Math.cos((minLat + maxLat) / 2 * Math.PI / 180);
    const scale = (MAP_WIDTH - 2 * MAP_PADDING) / ((maxLon - minLon) * lonScale);

    return {
        height: (maxLat - minLat) * scale + 2 * MAP_PADDING,
        toPoint: location => ({
            x: MAP_PADDING + (location.lon - minLon) * lonScale * scale,
            y: MAP_PADDING + (maxLat - location.lat) * scale
        })
    };
}

function addLabel(parent, location, point) {
    const offsets = {
        right: { x: 16, y: 5, anchor: "start" },
        left: { x: -16, y: 5, anchor: "end" },
        above: { x: 0, y: -19, anchor: "middle" },
        below: { x: 0, y: 25, anchor: "middle" },
        "below-right": { x: 8, y: 25, anchor: "start" }
    };
    const offset = offsets[location.label || "right"];
    const label = svgElement("text", {
        class: "map-label",
        x: point.x + offset.x,
        y: point.y + offset.y,
        "text-anchor": offset.anchor
    }, parent);

    label.textContent = location.name;
    return label;
}

function hospitalTooltip(location, hospitals) {
    const details = hospitals.map(hospital =>
        `${hospital.name}: ${hospital.specialties.join(", ").replace(/_/g, " ")}`
    );

    return [location.name, ...details].join("\n");
}

// Returns play() and stop(), so the caller decides when the search animation runs
function renderRouteMap(container, mapData, route) {
    const { height, toPoint } = createProjection(mapData.locations);
    const points = new Map(mapData.locations.map(location => [location.id, toPoint(location)]));
    const startId = route.path[0].id;
    const destinationId = route.path[route.path.length - 1].id;
    const pathIds = new Set(route.path.map(stop => stop.id));

    const svg = svgElement("svg", {
        class: "route-map",
        viewBox: `0 0 ${MAP_WIDTH} ${Math.ceil(height)}`,
        role: "img",
        "aria-label": `Map of Delhi NCR showing the route ${route.path.map(stop => stop.name).join(" to ")}.`
    });

    // Layers, drawn bottom to top
    const roadLayer = svgElement("g", {}, svg);
    const routeLayer = svgElement("g", {}, svg);
    const nodeLayer = svgElement("g", {}, svg);
    const labelLayer = svgElement("g", {}, svg);

    mapData.roads.forEach(road => {
        const from = points.get(road.from);
        const to = points.get(road.to);
        const isHeavy = road.congestion >= HEAVY_TRAFFIC_CONGESTION;

        svgElement("line", {
            class: isHeavy ? "map-road is-heavy" : "map-road",
            x1: from.x, y1: from.y, x2: to.x, y2: to.y
        }, roadLayer);
    });

    const routeSegments = route.path.slice(1).map((stop, index) => {
        const from = points.get(route.path[index].id);
        const to = points.get(stop.id);

        return svgElement("line", {
            class: "map-route",
            pathLength: 1,
            x1: from.x, y1: from.y, x2: to.x, y2: to.y
        }, routeLayer);
    });

    const nodes = new Map();
    const labels = new Map();

    mapData.locations.forEach(location => {
        const point = points.get(location.id);
        const hospitals = mapData.hospitals.filter(hospital => hospital.location === location.id);
        const canProvideCare = hospitals.some(hospital => hospital.specialties.includes(route.care));
        const group = svgElement("g", { class: "map-node" }, nodeLayer);

        svgElement("circle", { class: "map-explored", cx: point.x, cy: point.y, r: 17 }, group);

        if (hospitals.length > 0) {
            svgElement("rect", {
                class: canProvideCare ? "map-hospital is-suitable" : "map-hospital",
                x: point.x - 8, y: point.y - 8, width: 16, height: 16, rx: 3
            }, group);
        } else {
            svgElement("circle", { class: "map-dot", cx: point.x, cy: point.y, r: 5 }, group);
        }

        // A larger invisible target makes the tooltip easy to reach
        svgElement("circle", { class: "map-hit", cx: point.x, cy: point.y, r: 16 }, group);
        svgElement("title", {}, group).textContent = hospitals.length > 0
            ? hospitalTooltip(location, hospitals)
            : location.name;

        nodes.set(location.id, group);
        labels.set(location.id, addLabel(labelLayer, location, point));
    });

    nodes.get(startId).classList.add("is-start");
    labels.get(startId).classList.add("is-on-route");

    container.replaceChildren(svg);

    // Where the map is wider than its card, open it centred on the route
    const routeXs = route.path.map(stop => points.get(stop.id).x);
    const routeCentre = (Math.min(...routeXs) + Math.max(...routeXs)) / 2;
    const drawnScale = svg.getBoundingClientRect().width / MAP_WIDTH;
    container.scrollLeft = routeCentre * drawnScale - container.clientWidth / 2;

    const exploredIds = route.search.expansion_steps
        .map(step => step.node)
        .filter(id => id !== startId);
    let timers = [];

    function stop() {
        timers.forEach(clearTimeout);
        timers = [];
    }

    function reset() {
        exploredIds.forEach(id => nodes.get(id).classList.remove("is-explored"));
        routeSegments.forEach(segment => segment.classList.remove("is-drawn"));
        nodes.get(destinationId).classList.remove("is-destination");
        pathIds.forEach(id => {
            if (id !== startId) {
                labels.get(id).classList.remove("is-on-route");
            }
        });
    }

    function showExplored(id) {
        nodes.get(id).classList.add("is-explored");
    }

    function showRouteSegment(index) {
        routeSegments[index].classList.add("is-drawn");
        labels.get(route.path[index + 1].id).classList.add("is-on-route");
    }

    function showDestination() {
        nodes.get(destinationId).classList.add("is-destination");
        labels.get(destinationId).classList.add("is-on-route");
    }

    function play() {
        stop();
        reset();

        const steps = [
            ...exploredIds.map(id => [EXPLORE_STEP_MS, () => showExplored(id)]),
            ...routeSegments.map((segment, index) => [ROUTE_STEP_MS, () => showRouteSegment(index)]),
            [ROUTE_STEP_MS, showDestination]
        ];

        // Motion is only decoration here: the final picture is the same without it
        if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
            steps.forEach(([, step]) => step());
            return;
        }

        let delay = 0;

        steps.forEach(([duration, step]) => {
            timers.push(setTimeout(step, delay));
            delay += duration;
        });
    }

    return { play, stop };
}
