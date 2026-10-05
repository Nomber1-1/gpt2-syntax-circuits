"""
ACDC Circuit DAG Visualizer
===========================
Generates a Directed Acyclic Graph (DAG) of the discovered ACDC circuit.
Nodes are grouped by layer and colored by their performance drop.
"""

import json
from pathlib import Path
try:
    import graphviz
except ImportError:
    print("Error: The 'graphviz' library is not installed.")
    print("Please install it running: pip install graphviz")
    exit(1)

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "acdc_svn_results"
RESULTS_FILE = BASE / "acdc_results.json"

def main():
    if not RESULTS_FILE.exists():
        print(f"Error: Could not find results file at {RESULTS_FILE}")
        return

    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ablations = data.get("valid_ablations", [])
    circuit_nodes = [r for r in ablations if r.get("is_in_circuit")]
    
    if not circuit_nodes:
        print("No nodes found in the circuit. Ensure ACDC discovery saved correctly.")
        return

    # Group nodes by layer
    layers = sorted(list(set(n["layer"] for n in circuit_nodes)))
    
    # Initialize Graphviz Digraph
    dot = graphviz.Digraph(comment='ACDC SVN Circuit', format='png')
    dot.attr(rankdir='LR', splines='spline', nodesep='0.4', ranksep='1.0')
    
    dot.node('Input', 'Prompt\nInput', shape='box', style='filled', fillcolor='lightgrey', fontname='Helvetica-Bold')
    dot.node('Output', 'Logit\nPrediction', shape='box', style='filled', fillcolor='lightgrey', fontname='Helvetica-Bold')
    
    # Determine min and max drops for color intensity scaling
    drops = [n["performance_drop"] for n in circuit_nodes]
    min_drop, max_drop = min(drops), max(drops)
    
    nodes_by_layer = {}
    for l in layers:
        nodes_by_layer[l] = []
        # Group nodes in the same layer vertically
        with dot.subgraph(name=f'cluster_layer_{l}') as c:
            c.attr(label=f'Layer {l}', style='dashed', color='grey', fontname='Helvetica')
            for node in circuit_nodes:
                if node["layer"] == l:
                    head = node["head"]
                    drop = node["performance_drop"]
                    node_id = f'L{l}H{head}'
                    nodes_by_layer[l].append(node_id)
                    
                    # Calculate color intensity (light red to dark red)
                    intensity = int(255 * (drop - min_drop) / (max_drop - min_drop + 1e-5))
                    hex_color = f"#ff{255-intensity:02x}{255-intensity:02x}"
                    
                    c.node(node_id, f'L{l}H{head}\nDrop: {drop:.3f}', 
                           shape='ellipse', style='filled', fillcolor=hex_color, fontname='Helvetica')

    # Draw hierarchical edges connecting adjacent active layers
    for i in range(len(layers)):
        curr_l = layers[i]
        prev_nodes = ['Input'] if i == 0 else nodes_by_layer[layers[i-1]]
        next_nodes = ['Output'] if i == len(layers) - 1 else nodes_by_layer[layers[i+1]]
        
        # Connect current layer to next layer
        for n1 in nodes_by_layer[curr_l]:
            for n2 in next_nodes:
                dot.edge(n1, n2, color='#00000033')  # Highly transparent edges to avoid clutter
            if i == 0:
                for p in prev_nodes:
                    dot.edge(p, n1, color='#00000033')
                
    output_path = BASE / "acdc_circuit_dag"
    dot.render(str(output_path), view=False, cleanup=True)
    print(f"DAG visualization successfully saved to {output_path}.png")

if __name__ == "__main__":
    main()