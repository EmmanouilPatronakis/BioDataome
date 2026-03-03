#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import collections
import json

# ==========================================
# CONFIGURATION
# ==========================================
# Hardcode your directory path here
DATA_DIR = Path(r"./results")
OUTPUT_HTML = DATA_DIR / "interactive_cskl_network.html"


# ==========================================
# DATA LOADING & PROCESSING
# ==========================================
def load_data():
    print(f"Reading data from {DATA_DIR}...")

    # Load Network Edges
    edges_file = DATA_DIR / "cskl_network_edges.tsv"
    if not edges_file.exists():
        raise FileNotFoundError(f"Missing {edges_file}")
    edges_df = pd.read_csv(edges_file, sep="\t")

    # Load PCA Metadata
    meta_file = DATA_DIR / "pca_meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Missing {meta_file}")
    with open(meta_file, "r", encoding="utf-8") as f:
        pca_meta = json.load(f)

    # Load Edge Explainers
    explainers_file = DATA_DIR / "edge_explainers.json"
    if not explainers_file.exists():
        raise FileNotFoundError(f"Missing {explainers_file}")
    with open(explainers_file, "r", encoding="utf-8") as f:
        explainers = json.load(f)

    return edges_df, pca_meta, explainers


def build_network_data(edges_df, pca_meta, explainers, q_threshold=0.05):
    print("Building network data structures...")

    # 1. Filter globally for significance
    sig_edges = edges_df[edges_df["q_value"] <= q_threshold].copy()
    sig_edges = sig_edges.sort_values(["q_value", "cSKL"]).reset_index(drop=True)
    total_edges = len(sig_edges)

    # 2. Calculate Local Rank (Top K per dataset)
    # For every node, find all its edges and sort them by significance to assign a local rank.
    node_edges = collections.defaultdict(list)
    for idx, row in sig_edges.iterrows():
        node_edges[row["Dataset_A"]].append((row["Dataset_B"], row["q_value"], row["cSKL"], idx))
        node_edges[row["Dataset_B"]].append((row["Dataset_A"], row["q_value"], row["cSKL"], idx))

    edge_local_ranks = {}
    max_local_rank_observed = 1
    for node, edges in node_edges.items():
        # Sort by q_value, then cSKL
        edges.sort(key=lambda x: (x[1], x[2]))
        for local_rank, edge_info in enumerate(edges, 1):
            edge_idx = edge_info[3]
            # An edge's overall local rank is the minimum of its local rank on either connected node
            if edge_idx not in edge_local_ranks or local_rank < edge_local_ranks[edge_idx]:
                edge_local_ranks[edge_idx] = local_rank
            if local_rank > max_local_rank_observed:
                max_local_rank_observed = local_rank

    print(f"Found {total_edges} significant edges. Max connections on a single node: {max_local_rank_observed}.")

    nodes_data = []
    edges_data = []
    ui_explanations = {}
    ui_metadata = {}

    # 3. Process Nodes
    unique_nodes = set(sig_edges["Dataset_A"]).union(set(sig_edges["Dataset_B"]))
    for node in unique_nodes:
        nodes_data.append({"id": node, "label": node})
        meta = pca_meta.get(node, {})
        ui_metadata[node] = {
            "samples": meta.get("n_samples", "N/A"),
            "features": meta.get("n_features_used", "N/A"),
            "components": meta.get("c_components", "N/A"),
            "alpha": meta.get("alpha", "N/A")
        }

    # 4. Process Edges
    for idx, row in sig_edges.iterrows():
        source = row["Dataset_A"]
        target = row["Dataset_B"]
        q_val = row["q_value"]
        cskl_val = row["cSKL"]

        # Rankings
        global_rank = idx + 1
        global_percentile = (global_rank / total_edges) * 100
        local_rank = edge_local_ranks[idx]

        # Edge Width (1 to 8, stronger = thicker)
        width = max(1.0, 8.0 - (global_percentile / 100.0 * 7.0))

        edge_id_1 = f"{source}_{target}"
        edge_id_2 = f"{target}_{source}"
        edge_id = edge_id_1  # Default UI identifier

        edges_data.append({
            "id": edge_id,
            "from": source,
            "to": target,
            "width": width,
            "percentile": global_percentile,
            "local_rank": local_rank
        })

        # Lookup JSON Explainer
        exp_data = explainers.get(edge_id_1) or explainers.get(edge_id_2) or {}

        ui_explanations[edge_id] = {
            "cskl": f"{cskl_val:.4f}",
            "q_val": f"{q_val:.2e}",
            "global_rank": global_rank,
            "local_rank": local_rank,
            "total_computed": total_edges,
            "similar_features": exp_data.get("similar_features", []),
            "dissimilar_features": exp_data.get("dissimilar_features", [])
        }

    return nodes_data, edges_data, ui_metadata, ui_explanations, total_edges, max_local_rank_observed


# ==========================================
# HTML GENERATION
# ==========================================
def generate_html(nodes_data, edges_data, ui_metadata, ui_explanations, total_edges, max_local_rank):
    print("Generating HTML file...")

    num_nodes = len(nodes_data)
    # Default UI parameters
    default_mode = "local" if num_nodes < 100 else "global"
    default_local_val = 1

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>c-SKL Molecular Similarity Network</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        :root {{
            --bg-main: #f8fafc;
            --panel-bg: #ffffff;
            --text-dark: #1e293b;
            --text-muted: #64748b;
            --accent: #0ea5e9;
            --accent-hover: #0284c7;
            --border: #e2e8f0;
        }}
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; display: flex; height: 100vh; overflow: hidden; background-color: var(--bg-main); color: var(--text-dark); }}

        /* Layout */
        #left-panel {{ width: 320px; background: var(--panel-bg); padding: 25px 20px; box-shadow: 2px 0 15px rgba(0,0,0,0.03); z-index: 10; display: flex; flex-direction: column; border-right: 1px solid var(--border); overflow-y: auto; }}
        #network-container {{ flex-grow: 1; position: relative; }}
        #mynetwork {{ width: 100%; height: 100%; }}
        #right-panel {{ width: 420px; background: var(--panel-bg); padding: 25px 20px; box-shadow: -2px 0 15px rgba(0,0,0,0.03); overflow-y: auto; z-index: 10; border-left: 1px solid var(--border); transition: all 0.3s ease; }}

        /* Typography */
        h1, h2, h3 {{ margin-top: 0; color: var(--text-dark); }}
        h1 {{ font-size: 1.25rem; margin-bottom: 5px; font-weight: 700; }}
        h2 {{ font-size: 1.05rem; border-bottom: 2px solid var(--accent); padding-bottom: 8px; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 0.5px; }}
        h3 {{ font-size: 0.95rem; margin-top: 25px; margin-bottom: 10px; color: var(--text-muted); font-weight: 600; }}
        p {{ font-size: 0.9rem; line-height: 1.5; color: var(--text-muted); }}

        /* Stats & Cards */
        .stat-box {{ background: var(--bg-main); border: 1px solid var(--border); padding: 15px; border-radius: 8px; margin-bottom: 20px; font-size: 0.9rem; }}
        .stat-row {{ display: flex; justify-content: space-between; margin-bottom: 8px; border-bottom: 1px dashed var(--border); padding-bottom: 4px; }}
        .stat-row:last-child {{ margin-bottom: 0; border-bottom: none; padding-bottom: 0; }}
        .stat-label {{ color: var(--text-muted); font-weight: 500; }}
        .stat-value {{ font-weight: 600; color: var(--text-dark); text-align: right; }}

        /* Controls */
        .control-group {{ margin-bottom: 25px; background: var(--panel-bg); }}
        label {{ font-size: 0.85rem; font-weight: 600; display: block; margin-bottom: 10px; color: var(--text-dark); display: flex; justify-content: space-between; align-items: center; }}

        /* Radio Buttons */
        .radio-group {{ display: flex; flex-direction: column; gap: 8px; margin-bottom: 15px; background: var(--bg-main); padding: 12px; border-radius: 6px; border: 1px solid var(--border); }}
        .radio-item {{ display: flex; align-items: center; gap: 8px; font-size: 0.9rem; }}

        button {{ background: var(--accent); color: white; border: none; padding: 10px 15px; cursor: pointer; border-radius: 6px; width: 100%; font-weight: 600; transition: background 0.2s, transform 0.1s; box-shadow: 0 2px 4px rgba(14, 165, 233, 0.2); }}
        button:hover {{ background: var(--accent-hover); }}
        button:active {{ transform: scale(0.98); }}

        /* Slider */
        input[type=range] {{ width: 100%; margin-top: 5px; accent-color: var(--accent); cursor: pointer; }}
        .slider-val-badge {{ background: var(--accent); color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }}

        /* Lists for Features */
        ul {{ list-style-type: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 6px; }}
        li.feat-item {{ display: flex; justify-content: space-between; background: var(--bg-main); border: 1px solid var(--border); padding: 8px 12px; border-radius: 4px; font-size: 0.85rem; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; align-items: center; }}
        .feat-name {{ color: var(--accent-hover); font-weight: 600; }}
        .feat-score {{ color: var(--text-muted); font-size: 0.75rem; background: #e2e8f0; padding: 2px 6px; border-radius: 4px; }}

        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: #94a3b8; }}

        .badge-significance {{ background: #dcfce7; color: #166534; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; border: 1px solid #bbf7d0; }}
    </style>
</head>
<body>

    <div id="left-panel">
        <h1>c-SKL Network</h1>
        <p style="margin-bottom: 25px; font-size: 0.8rem;">Visualizing statistically significant dataset similarities.</p>

        <h2>Filters & Controls</h2>

        <div class="control-group">
            <label>Filtering Mode</label>
            <div class="radio-group">
                <div class="radio-item">
                    <input type="radio" id="modeLocal" name="filterMode" value="local" onchange="switchMode()" {"checked" if default_mode == "local" else ""}>
                    <label for="modeLocal" style="margin:0; font-weight:normal;">Top K Links per Dataset</label>
                </div>
                <div class="radio-item">
                    <input type="radio" id="modeGlobal" name="filterMode" value="global" onchange="switchMode()" {"checked" if default_mode == "global" else ""}>
                    <label for="modeGlobal" style="margin:0; font-weight:normal;">Top % Globally</label>
                </div>
            </div>
        </div>

        <div class="control-group">
            <label>
                <span id="slider-title">Filter Limit</span>
                <span class="slider-val-badge" id="slider-val-display"></span>
            </label>
            <input type="range" id="edgeSlider" oninput="updateFilter()">
            <p id="slider-desc" style="font-size: 0.75rem; margin-top: 8px;"></p>
        </div>

        <div class="control-group">
            <label>Physics Engine</label>
            <button onclick="togglePhysics()" id="physicsBtn">Freeze Layout</button>
        </div>

        <div class="stat-box">
            <div class="stat-row"><span class="stat-label">Total Datasets</span><span class="stat-value">{num_nodes}</span></div>
            <div class="stat-row"><span class="stat-label">Significant Edges</span><span class="stat-value">{total_edges}</span></div>
        </div>
    </div>

    <div id="network-container">
        <div id="mynetwork"></div>
    </div>

    <div id="right-panel">
        <h2>Inspector</h2>

        <div id="default-info">
            <p style="background: var(--bg-main); padding: 15px; border-radius: 8px; border: 1px dashed var(--border);">
                <strong>Interactive Mode:</strong><br><br>
                🟢 Click on a <strong>Node</strong> to view dataset architecture.<br><br>
                🔗 Click on an <strong>Edge</strong> to view the molecular features driving the similarity.
            </p>
        </div>

        <div id="node-info" style="display: none;">
            <h3>Dataset Selected</h3>
            <h1 id="node-title" style="color: var(--accent); margin-bottom: 15px; font-family: monospace; font-size: 1.4rem;"></h1>
            <div class="stat-box">
                <div class="stat-row"><span class="stat-label">Samples (m)</span><span class="stat-value" id="node-samples"></span></div>
                <div class="stat-row"><span class="stat-label">Features (n)</span><span class="stat-value" id="node-features"></span></div>
                <div class="stat-row"><span class="stat-label">PCA Components (c)</span><span class="stat-value" id="node-comps"></span></div>
                <div class="stat-row"><span class="stat-label">Alpha (Variance)</span><span class="stat-value" id="node-alpha"></span></div>
            </div>
        </div>

        <div id="edge-info" style="display: none;">
            <h3>Similarity Connection</h3>
            <h1 id="edge-title" style="color: var(--accent); font-size: 1.1rem; margin-bottom: 15px; font-family: monospace;"></h1>
            <div class="stat-box">
                <div class="stat-row"><span class="stat-label">Local Rank</span><span class="stat-value">Top <span id="edge-local-rank"></span> for node</span></div>
                <div class="stat-row"><span class="stat-label">Global Rank</span><span class="stat-value">#<span id="edge-global-rank"></span> of <span id="edge-total"></span></span></div>
                <div class="stat-row"><span class="stat-label">c-SKL Distance</span><span class="stat-value" id="edge-cskl"></span></div>
                <div class="stat-row"><span class="stat-label">FDR q-value</span><span class="stat-value badge-significance" id="edge-qval"></span></div>
            </div>

            <h3>Top Similar Features (B(k))</h3>
            <ul id="list-similar"></ul>

            <h3 style="margin-top: 30px;">Top Dissimilar Features (W(k))</h3>
            <ul id="list-dissimilar"></ul>
        </div>
    </div>

    <script>
        // 1. Data Injection
        const nodesData = {json.dumps(nodes_data)};
        const allEdgesData = {json.dumps(edges_data)};
        const explanations = {json.dumps(ui_explanations)};
        const nodeMetadata = {json.dumps(ui_metadata)};
        const maxLocalRank = {max_local_rank};

        // 2. Vis.js Initialization
        const nodesDataset = new vis.DataSet(nodesData);
        const edgesDataset = new vis.DataSet(allEdgesData);

        let currentMode = "{default_mode}";
        let currentLimit = currentMode === "local" ? {default_local_val} : 100;

        const edgesFilter = (item) => {{
            if (currentMode === "local") {{
                return item.local_rank <= currentLimit;
            }} else {{
                return item.percentile <= currentLimit;
            }}
        }};

        const edgesView = new vis.DataView(edgesDataset, {{ filter: edgesFilter }});

        const container = document.getElementById('mynetwork');
        const data = {{ nodes: nodesDataset, edges: edgesView }};

        const options = {{
            nodes: {{ 
                shape: 'dot', 
                size: 18, 
                color: {{ background: '#0ea5e9', border: '#0284c7', highlight: {{ background: '#f59e0b', border: '#b45309' }} }},
                font: {{ size: 13, face: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto', color: '#1e293b' }},
                borderWidth: 2, borderWidthSelected: 3
            }},
            edges: {{ 
                color: {{ color: '#cbd5e1', highlight: '#f59e0b', hover: '#94a3b8' }}, 
                smooth: {{ type: 'continuous' }}, hoverWidth: 2
            }},
            physics: {{ 
                barnesHut: {{ gravitationalConstant: -3000, centralGravity: 0.2, springLength: 180, springConstant: 0.04 }},
                stabilization: {{ iterations: 200 }}
            }},
            interaction: {{ hover: true, tooltipDelay: 200, zoomSpeed: 0.5 }}
        }};

        const network = new vis.Network(container, data, options);

        // 3. UI Interactions
        function initUI() {{
            switchMode(); // Set correct limits and labels on load
        }}

        function switchMode() {{
            const isLocal = document.getElementById('modeLocal').checked;
            currentMode = isLocal ? "local" : "global";

            const slider = document.getElementById('edgeSlider');
            if (currentMode === "local") {{
                slider.min = 1;
                slider.max = maxLocalRank;
                slider.value = {default_local_val};
                document.getElementById('slider-title').innerText = "Links per Dataset";
                document.getElementById('slider-desc').innerText = "Guarantees up to K edges are shown for every dataset.";
            }} else {{
                slider.min = 1;
                slider.max = 100;
                slider.value = 100;
                document.getElementById('slider-title').innerText = "Global Percentile";
                document.getElementById('slider-desc').innerText = "Shows the top X% of the absolute strongest edges in the entire network.";
            }}
            updateFilter();
        }}

        function updateFilter() {{
            const slider = document.getElementById('edgeSlider');
            currentLimit = parseInt(slider.value, 10);
            document.getElementById('slider-val-display').innerText = currentMode === "local" ? currentLimit : currentLimit + "%";
            edgesView.refresh();
        }}

        let physicsEnabled = true;
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{ physics: physicsEnabled }});
            document.getElementById('physicsBtn').innerText = physicsEnabled ? "Unfreeze Layout" : "Freeze Layout";
            document.getElementById('physicsBtn').style.background = physicsEnabled ? "var(--accent)" : "#ef4444";
        }}

        // Format array of objects into HTML list
        function formatFeatureList(features) {{
            if (!Array.isArray(features) || features.length === 0) return `<li class="feat-item"><span class="feat-name">No data calculated</span></li>`;
            
            return features.map(f => {{
                // Handle dict/object format {{"gene": "...", "score": ...}}
                if (typeof f === 'object' && f !== null) {{
                    const gene = f.gene || 'Unknown';
                    const score = f.score !== undefined ? Number(f.score).toExponential(2) : 'N/A';
                    return `<li class="feat-item"><span class="feat-name">${{gene}}</span> <span class="feat-score">${{score}}</span></li>`;
                }}
                // Fallback for flat lists
                return `<li class="feat-item"><span class="feat-name">${{f}}</span></li>`;
            }}).join('');
        }}

        // 4. Click Handling Logic
        network.on("click", function (params) {{
            const defaultInfo = document.getElementById('default-info');
            const nodeInfo = document.getElementById('node-info');
            const edgeInfo = document.getElementById('edge-info');

            defaultInfo.style.display = 'none';
            nodeInfo.style.display = 'none';
            edgeInfo.style.display = 'none';

            if (params.nodes.length > 0) {{
                // NODE CLICKED
                const nodeId = params.nodes[0];
                const meta = nodeMetadata[nodeId];
                if (meta) {{
                    nodeInfo.style.display = 'block';
                    document.getElementById('node-title').innerText = nodeId;
                    document.getElementById('node-samples').innerText = meta.samples;
                    document.getElementById('node-features').innerText = meta.features;
                    document.getElementById('node-comps').innerText = meta.components;
                    document.getElementById('node-alpha').innerText = meta.alpha;
                }}
            }} 
            else if (params.edges.length > 0) {{
                // EDGE CLICKED
                const edgeId = params.edges[0];
                const exp = explanations[edgeId];
                const edgeObj = edgesDataset.get(edgeId); 

                if (exp && edgeObj) {{
                    edgeInfo.style.display = 'block';
                    document.getElementById('edge-title').innerHTML = `${{edgeObj.from}} <span style="color:#94a3b8;">↔</span> ${{edgeObj.to}}`;
                    document.getElementById('edge-local-rank').innerText = exp.local_rank;
                    document.getElementById('edge-global-rank').innerText = exp.global_rank;
                    document.getElementById('edge-total').innerText = exp.total_computed;
                    document.getElementById('edge-cskl').innerText = exp.cskl;
                    document.getElementById('edge-qval').innerText = exp.q_val;

                    document.getElementById('list-similar').innerHTML = formatFeatureList(exp.similar_features);
                    document.getElementById('list-dissimilar').innerHTML = formatFeatureList(exp.dissimilar_features);
                }}
            }} 
            else {{
                // EMPTY CANVAS
                defaultInfo.style.display = 'block';
            }}
        }});

        // Trigger UI setup
        initUI();
    </script>
</body>
</html>"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"\nSuccess! UI Generated.")
    print(f"Open this file in Google Chrome or Edge: {OUTPUT_HTML}")


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        edges_df, pca_meta, explainers = load_data()
        nodes_data, edges_data, ui_metadata, ui_explanations, total_edges, max_local = build_network_data(edges_df,
                                                                                                          pca_meta,
                                                                                                          explainers)
        generate_html(nodes_data, edges_data, ui_metadata, ui_explanations, total_edges, max_local)
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        print("Please ensure DATA_DIR is correct and required files exist.")
