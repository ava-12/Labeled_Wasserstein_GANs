import os
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import (
    scatter,
    degree,
    to_networkx
)

import itertools
import random
import networkx as nx
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


# --------------------------
#  Dataset Statistics Helper
# --------------------------
from collections import defaultdict
import numpy as np

def analyze_dataset_statistics(dataset):
    """Analyze node count statistics per class and infer num_classes automatically."""
    class_stats = defaultdict(list)
    
    # Collect node counts per class
    for data in dataset:
        class_label = int(data.y.item())
        num_nodes = data.x.size(0)
        class_stats[class_label].append(num_nodes)
    
    # Infer num_classes from observed class labels
    num_classes = max(class_stats.keys()) + 1
    
    stats_summary = {}
    for class_id in range(num_classes):
        if class_id in class_stats:
            node_counts = class_stats[class_id]
            stats_summary[class_id] = {
                'min': min(node_counts),
                'max': max(node_counts),
                'mean': np.mean(node_counts),
                'std': np.std(node_counts),
                'median': np.median(node_counts),
                'count': len(node_counts)
            }
            print(f"Class {class_id}: {len(node_counts)} graphs, "
                  f"nodes range: {min(node_counts)}-{max(node_counts)}, "
                  f"mean: {np.mean(node_counts):.1f}±{np.std(node_counts):.1f}")
    
    return stats_summary



# --------------------------
#  Evaluation Metrics Functions
# --------------------------

def graph_to_features(data):
    """Extract structural features from a graph for MMD computation"""
    edge_index = data.edge_index.cpu().numpy()
    num_nodes = data.x.size(0)
    
    if edge_index.shape[1] == 0:  # No edges
        return np.array([num_nodes, 0, 0, 0, 0, 0])
    
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.T)
    
    # Structural features
    num_edges = G.number_of_edges()
    density = nx.density(G)
    
    # Degree statistics
    degrees = list(dict(G.degree()).values())
    avg_degree = np.mean(degrees) if degrees else 0
    max_degree = np.max(degrees) if degrees else 0
    
    # Clustering coefficient
    try:
        clustering = nx.average_clustering(G)
    except:
        clustering = 0
    
    return np.array([num_nodes, num_edges, density, avg_degree, max_degree, clustering])

def compute_mmd(X, Y, kernel='rbf', gamma=1.0):
    """Compute Maximum Mean Discrepancy between two sets of samples"""
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    
    m = X.size(0)
    n = Y.size(0)
    
    if kernel == 'rbf':
        # RBF kernel
        XX = torch.mm(X, X.t())
        YY = torch.mm(Y, Y.t())
        XY = torch.mm(X, Y.t())
        
        X_sqnorms = torch.diag(XX).unsqueeze(1)
        Y_sqnorms = torch.diag(YY).unsqueeze(1)
        
        K_XX = torch.exp(-gamma * (X_sqnorms + X_sqnorms.t() - 2 * XX))
        K_YY = torch.exp(-gamma * (Y_sqnorms + Y_sqnorms.t() - 2 * YY))
        K_XY = torch.exp(-gamma * (X_sqnorms + Y_sqnorms.t() - 2 * XY))
        
    elif kernel == 'linear':
        K_XX = torch.mm(X, X.t())
        K_YY = torch.mm(Y, Y.t())
        K_XY = torch.mm(X, Y.t())
    
    # MMD computation
    mmd = (K_XX.sum() / (m * m) + K_YY.sum() / (n * n) - 2 * K_XY.sum() / (m * n))
    
    return mmd.item()

def graph_to_canonical_string(data):
    """Convert graph to a canonical string representation for uniqueness/novelty checks"""
    edge_index = data.edge_index.cpu().numpy()
    num_nodes = data.x.size(0)
    
    # Create adjacency matrix
    adj = np.zeros((num_nodes, num_nodes))
    if edge_index.shape[1] > 0:
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1  # Undirected
    
    # Get canonical form
    G = nx.from_numpy_array(adj)
    
    try:
        # Simple canonical form based on degree sequence and edges
        degree_seq = sorted([d for n, d in G.degree()])
        edges = sorted(list(G.edges()))
        canonical = f"nodes:{num_nodes}_degrees:{degree_seq}_edges:{edges}"
        return canonical
    except:
        return f"nodes:{num_nodes}_isolated"

def compute_uniqueness(generated_graphs):
    """Compute uniqueness of generated graphs"""
    if not generated_graphs:
        return 0.0
    
    canonical_forms = set()
    for graph in generated_graphs:
        canonical = graph_to_canonical_string(graph)
        canonical_forms.add(canonical)
    
    return len(canonical_forms) / len(generated_graphs)

def compute_novelty(generated_graphs, training_graphs):
    """Compute novelty of generated graphs compared to training set"""
    if not generated_graphs or not training_graphs:
        return 0.0
    
    # Get canonical forms of training graphs
    training_canonical = set()
    for graph in training_graphs:
        canonical = graph_to_canonical_string(graph)
        training_canonical.add(canonical)
    
    # Check how many generated graphs are novel
    novel_count = 0
    for graph in generated_graphs:
        canonical = graph_to_canonical_string(graph)
        if canonical not in training_canonical:
            novel_count += 1
    
    return novel_count / len(generated_graphs)

def compute_validity(generated_graphs, min_nodes=3, max_nodes=50):
    """Compute validity of generated graphs based on basic constraints"""
    if not generated_graphs:
        return 0.0
    
    valid_count = 0
    for graph in generated_graphs:
        num_nodes = graph.x.size(0)
        num_edges = graph.edge_index.size(1)
        
        # Basic validity checks
        is_valid = True
        
        # Check node count
        if num_nodes < min_nodes or num_nodes > max_nodes:
            is_valid = False
        
        # Check for self-loops (should not have any)
        if num_edges > 0:
            edge_index = graph.edge_index
            if (edge_index[0] == edge_index[1]).any():
                is_valid = False
        
        # Check for reasonable edge count (not too dense)
        max_edges = num_nodes * (num_nodes - 1) // 2
        if num_edges > max_edges:
            is_valid = False
        
        if is_valid:
            valid_count += 1
    
    return valid_count / len(generated_graphs)

def extract_individual_graphs(batch_data, device):
    """
    Extract individual graphs from a batched PyG graph object.
    Preserves node features, edges, and graph-level labels (batch_data.y).
    """
    batch_data = batch_data.to(device)
    individual_graphs = []

    # If no batch attribute, just return single graph
    if not hasattr(batch_data, 'batch'):
        return [batch_data]

    batch = batch_data.batch
    num_graphs = int(batch.max().item()) + 1  # number of graphs in batch

    for i in range(num_graphs):
        node_mask = (batch == i)
        node_features = batch_data.x[node_mask]

        # Extract edges that connect nodes within this graph
        edge_mask = node_mask[batch_data.edge_index[0]] & node_mask[batch_data.edge_index[1]]
        if edge_mask.any():
            edges = batch_data.edge_index[:, edge_mask]

            # Renumber nodes from 0
            unique_nodes = torch.unique(edges)
            mapping = {node.item(): idx for idx, node in enumerate(unique_nodes)}
            renumbered_edges = torch.tensor([[mapping[e[0].item()], mapping[e[1].item()]] 
                                             for e in edges.t()]).t().to(device)
        else:
            renumbered_edges = torch.empty((2, 0), dtype=torch.long, device=device)

        # Extract graph-level label if available
        graph_label = None
        if hasattr(batch_data, 'y') and batch_data.y is not None:
            y = batch_data.y
            if y.ndim == 1 or y.shape[0] == num_graphs:
                graph_label = y[i].unsqueeze(0)
            else:
                graph_label = y[i]

        individual_graphs.append(Data(x=node_features, edge_index=renumbered_edges, y=graph_label))

    return individual_graphs

# ------------------------------
# Helper functions for graph statistics
# ------------------------------

def degree_distribution(graph, num_bins=10):
    """
    Compute degree distribution of a PyG graph.
    Returns a histogram of degrees (normalized to sum=1).
    """
    if graph.num_nodes == 0:
        return np.zeros(num_bins, dtype=float)
    
    row, col = graph.edge_index
    deg = degree(row, num_nodes=graph.num_nodes)  # torch tensor

    # Convert to numpy
    deg = deg.cpu().numpy()
    
    # Use bins from min to max degree
    hist, bin_edges = np.histogram(deg, bins=num_bins, range=(deg.min(), deg.max()), density=True)
    
    # Ensure normalized sum = 1
    if hist.sum() > 0:
        hist = hist / hist.sum()
    
    return hist

def clustering_distribution(graph):
    # Compute clustering coefficients
    clust_coeffs = np.array(list(nx.clustering(graph).values()))
    
    # Compute normalized histogram
    hist, _ = np.histogram(clust_coeffs, bins=10, range=(0, 1), density=True)
    return hist

def node_label_distribution(graph, num_classes=None):
    """
    Compute the normalized histogram of node labels for a PyG Data object.
    Ensures fixed-length output for MMD computation.
    
    Args:
        graph: PyG Data object with graph.y (node labels)
        num_classes: total number of classes in dataset
    Returns:
        hist: np.array of length num_classes, normalized
    """
    if not hasattr(graph, 'y') or graph.y is None:
        # return uniform zero histogram if no labels
        return np.zeros(num_classes if num_classes is not None else 1, dtype=float)

    labels = graph.y.cpu().numpy().flatten()
    
    if num_classes is None:
        num_classes = labels.max() + 1  # assume labels start at 0

    hist = np.zeros(num_classes, dtype=float)
    for l in labels:
        hist[l] += 1

    # Normalize to sum=1
    if hist.sum() > 0:
        hist /= hist.sum()

    return hist

def orbit_distribution(graph):
    """Return normalized orbit features (here: triangle density)."""
    num_triangles = sum(nx.triangles(graph).values()) / 3  # each triangle counted 3 times
    n_nodes = graph.number_of_nodes()
    
    if n_nodes < 3:
        return np.array([0.0])
    
    # Max possible triangles in an n-node graph = C(n, 3)
    max_triangles = n_nodes * (n_nodes - 1) * (n_nodes - 2) / 6
    tri_density = num_triangles / max_triangles if max_triangles > 0 else 0.0
    
    return np.array([tri_density])

def extract_graph_statistics(graphs, num_classes):
    """
    Extracts degree, clustering, node label, and orbit distributions from a list of graphs.
    All outputs are aligned in dimension for MMD computation.
    """
    degree_dists, clustering_dists, label_dists, orbit_dists = [], [], [], []

    for g in graphs:
        # Degree distribution (normalized histogram)
        degree_dists.append(degree_distribution(g))

        # Label distribution (fixed-length histogram)
        label_dists.append(node_label_distribution(g, num_classes=num_classes))

        # Convert to NetworkX
        G_nx = to_networkx(g, to_undirected=True)

        # Orbit distribution
        orbit_dists.append(orbit_distribution(G_nx))

        # Clustering distribution
        clustering_dists.append(clustering_distribution(G_nx))

    return (
        np.array(degree_dists),
        np.array(clustering_dists),
        np.array(label_dists),
        np.array(orbit_dists),
    )


def evaluate_model(generator, test_data_loader, train_data_loader, device, dataset_stats, num_classes, num_samples=500):
    """Comprehensive evaluation of the generator model using test set.
    Computes validity, uniqueness, novelty, and MMD for degree, clustering, node labels, and orbit counts.
    """
    generator.eval()
    print("Starting model evaluation...")
    
    # ------------------------------
    # Collect test graphs
    # ------------------------------
    test_graphs = []
    for batch in test_data_loader:
        batch = batch.to(device)
        test_graphs.extend(extract_individual_graphs(batch, device))
        if len(test_graphs) >= num_samples:
            break
    test_graphs = test_graphs[:num_samples]
    print(f"Collected {len(test_graphs)} test graphs.")


    # ------------------------------
    # Collect train graphs for novelty
    # ------------------------------
    train_graphs = []
    for batch in train_data_loader:
        batch = batch.to(device)
        train_graphs.extend(extract_individual_graphs(batch, device))
        if len(train_graphs) >= 1000:  # Limit for efficiency
            break
    print(f"Collected {len(train_graphs)} training graphs for novelty computation.")

    # ------------------------------
    # Generate fake graphs
    # ------------------------------
    generated_graphs = []
    with torch.no_grad():
        samples_per_batch = 16
        num_batches = (num_samples + samples_per_batch - 1) // samples_per_batch
        for batch_idx in range(num_batches):
            current_batch_size = min(samples_per_batch, num_samples - batch_idx * samples_per_batch)
            class_labels = torch.randint(0, 2, (current_batch_size,)).to(device)
            fake_batch = generator(num_graphs=current_batch_size, class_labels=class_labels, dataset_stats=dataset_stats)
            fake_batch = fake_batch.to(device)
            generated_graphs.extend(extract_individual_graphs(fake_batch, device))
    generated_graphs = generated_graphs[:num_samples]
    print(f"Generated {len(generated_graphs)} graphs.")

    # ------------------------------
    # Compute MMD for each graph statistic
    # ------------------------------
    test_degree, test_clust, test_label, test_orbit = extract_graph_statistics(test_graphs, num_classes)
    gen_degree, gen_clust, gen_label, gen_orbit = extract_graph_statistics(generated_graphs, num_classes)

    mmd_degree = compute_mmd(test_degree, gen_degree, kernel='rbf', gamma=1.0)
    mmd_clust  = compute_mmd(test_clust, gen_clust, kernel='rbf', gamma=1.0)
    mmd_label  = compute_mmd(test_label, gen_label, kernel='rbf', gamma=1.0)
    mmd_orbit  = compute_mmd(test_orbit, gen_orbit, kernel='rbf', gamma=1.0)

    print(f"MMD Degree: {mmd_degree:.4f}")
    print(f"MMD Clustering: {mmd_clust:.4f}")
    print(f"MMD Node Labels: {mmd_label:.4f}")
    print(f"MMD Orbit Counts: {mmd_orbit:.4f}")

    # ------------------------------
    # Compute validity, uniqueness, novelty
    # ------------------------------
    # Uniqueness
    uniqueness = compute_uniqueness(generated_graphs)
    
    # Novelty (comparing against TRAINING set)
    novelty = compute_novelty(generated_graphs, train_graphs)
    
    # Validity
    validity = compute_validity(generated_graphs)
    
    # Additional statistics
    test_stats = {
        'num_nodes': np.mean([g.x.size(0) for g in test_graphs]),
        'num_edges': np.mean([g.edge_index.size(1) for g in test_graphs]),
        'density': np.mean([graph_to_features(g)[2] for g in test_graphs])
    }
    
    gen_stats = {
        'num_nodes': np.mean([g.x.size(0) for g in generated_graphs]),
        'num_edges': np.mean([g.edge_index.size(1) for g in generated_graphs]),
        'density': np.mean([graph_to_features(g)[2] for g in generated_graphs])
    }
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Uniqueness:           {uniqueness:.4f}")
    print(f"MMD Degree:         {mmd_degree:.4f}")
    print(f"MMD Clustering:     {mmd_clust:.4f}")
    print(f"MMD Node Labels:     {mmd_label:.4f}")
    print(f"MMD Orbit Counts:     {mmd_orbit:.4f}")
    print(f"Novelty:              {novelty:.4f}")
    print(f"Validity:             {validity:.4f}")
    print("\nGraph Statistics Comparison:")
    print(f"Average Nodes  - Test: {test_stats['num_nodes']:.2f}, Generated: {gen_stats['num_nodes']:.2f}")
    print(f"Average Edges  - Test: {test_stats['num_edges']:.2f}, Generated: {gen_stats['num_edges']:.2f}")
    print(f"Average Density - Test: {test_stats['density']:.4f}, Generated: {gen_stats['density']:.4f}")
    print("="*50)
    
    return {
        "mmd_degree": mmd_degree,
        "mmd_clust": mmd_clust,
        "mmd_label": mmd_label,
        "mmd_orbit": mmd_orbit,
        'uniqueness': uniqueness,
        'novelty': novelty,
        'validity': validity,
        'test_stats': test_stats,
        'generated_stats': gen_stats
    }


def evaluate_class_specific(generator, test_data_loader, train_data_loader, device,
                            dataset_stats, num_classes, target_class=0, num_samples=150):
    """Evaluate generator for a specific class, compute MMDs, uniqueness, novelty, validity."""
    generator.eval()
    print(f"\nEvaluating Class {target_class} specifically...")

    # ------------------------------
    # Generate graphs of target class
    # ------------------------------
    generated_graphs_class = []
    with torch.no_grad():
        samples_per_batch = 16
        num_batches = (num_samples + samples_per_batch - 1) // samples_per_batch
        for batch_idx in range(num_batches):
            current_batch_size = min(samples_per_batch, num_samples - batch_idx * samples_per_batch)
            class_labels = torch.full((current_batch_size,), target_class, device=device)
            fake_batch = generator(num_graphs=current_batch_size, class_labels=class_labels,
                                   dataset_stats=dataset_stats)
            fake_batch = fake_batch.to(device)
            generated_graphs_class.extend(extract_individual_graphs(fake_batch, device))
    
    generated_graphs_class = generated_graphs_class[:num_samples]
    print(f"Generated {len(generated_graphs_class)} graphs for class {target_class}")

    # ------------------------------
    # Collect test graphs of target class
    # ------------------------------
    test_graphs_class = []
    for batch in test_data_loader:
        batch = batch.to(device)
        if batch.y is None:
            continue

        graph_indices = torch.unique(batch.batch)
        for graph_idx in graph_indices:
            if batch.y[graph_idx] != target_class:
                continue

            node_mask = (batch.batch == graph_idx)
            node_features = batch.x[node_mask]

            # Edges within this graph
            if batch.edge_index.size(1) > 0:
                edge_mask = node_mask[batch.edge_index[0]] & node_mask[batch.edge_index[1]]
                if edge_mask.any():
                    edges = batch.edge_index[:, edge_mask]
                    unique_nodes = torch.unique(edges)
                    mapping = {n.item(): i for i, n in enumerate(unique_nodes)}
                    renumbered_edges = torch.tensor([[mapping[e[0].item()], mapping[e[1].item()]]
                                                     for e in edges.t()]).t()
                else:
                    renumbered_edges = torch.empty((2,0), dtype=torch.long)
            else:
                renumbered_edges = torch.empty((2,0), dtype=torch.long)

            graph_data = Data(x=node_features, edge_index=renumbered_edges,
                              y=batch.y[graph_idx].unsqueeze(0))
            test_graphs_class.append(graph_data)
        if len(test_graphs_class) >= num_samples:
            break
    print(f"Collected {len(test_graphs_class)} test graphs for class {target_class}")

    # ------------------------------
    # Collect training graphs for novelty
    # ------------------------------
    train_graphs_class = []
    for batch in train_data_loader:
        batch = batch.to(device)
        if batch.y is None:
            continue
        mask = (batch.y == target_class)
        if mask.any():
            train_graphs_class.extend(extract_individual_graphs(batch, device))

    # ------------------------------
    # Compute graph statistics
    # ------------------------------
    test_degree, test_clust, test_label, test_orbit = extract_graph_statistics(test_graphs_class, num_classes)
    gen_degree, gen_clust, gen_label, gen_orbit = extract_graph_statistics(generated_graphs_class, num_classes)

    # Compute MMDs
    mmd_degree = compute_mmd(test_degree, gen_degree, kernel='rbf')
    mmd_clust  = compute_mmd(test_clust, gen_clust, kernel='rbf')
    mmd_label  = compute_mmd(test_label, gen_label, kernel='rbf')
    mmd_orbit  = compute_mmd(test_orbit, gen_orbit, kernel='rbf')

    # ------------------------------
    # Other metrics
    # ------------------------------
    uniqueness_class = compute_uniqueness(generated_graphs_class)
    validity_class = compute_validity(generated_graphs_class)
    novelty_class = compute_novelty(generated_graphs_class, train_graphs_class)

    # Print results
    print(f"Class {target_class} Evaluation:")
    print(f"MMD Degree: {mmd_degree:.4f}, Clustering: {mmd_clust:.4f}, Labels: {mmd_label:.4f}, Orbit: {mmd_orbit:.4f}")
    print(f"Uniqueness: {uniqueness_class:.4f}, Novelty: {novelty_class:.4f}, Validity: {validity_class:.4f}")

    return {
        'mmd_degree': mmd_degree,
        'mmd_clust': mmd_clust,
        'mmd_label': mmd_label,
        'mmd_orbit': mmd_orbit,
        'uniqueness': uniqueness_class,
        'novelty': novelty_class,
        'validity': validity_class,
        'num_test': len(test_graphs_class),
        'num_generated': len(generated_graphs_class)
    }

def visualize_generated_graphs(generator, device, dataset_stats, num_samples=6):
    """Generate and visualize multiple graphs"""
    print(f"\nVisualizing {num_samples} generated graphs...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i in range(num_samples):
            label = torch.tensor([i % 2]).to(device)  # Alternate between classes
            fake = generator(num_graphs=1, class_labels=label, dataset_stats=dataset_stats)
            
            edge_index = fake.edge_index.cpu().numpy()
            num_nodes = fake.x.size(0)
            
            if edge_index.shape[1] > 0:
                G = nx.Graph()
                G.add_nodes_from(range(num_nodes))
                G.add_edges_from(edge_index.T)
                
                # Color nodes by class
                node_colors = ['lightblue' if label.item() == 0 else 'lightcoral'] * num_nodes
                
                nx.draw(G, ax=axes[i], node_size=100, with_labels=True, 
                       node_color=node_colors, font_size=8)
                axes[i].set_title(f"Graph {i+1} (Class {label.item()})\nNodes: {num_nodes}, Edges: {len(G.edges())}")
            else:
                axes[i].text(0.5, 0.5, f'No edges\n{num_nodes} nodes', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"Graph {i+1} (Class {label.item()})")
    
    plt.tight_layout()
    plt.show()

# --------------------------
#  Main Evaluation Script
# --------------------------

def run_complete_evaluation(generator, train_loader, test_loader, device, dataset_stats, num_classes):
    """Run complete evaluation of a trained model"""
    
    print("\n" + "="*80)
    print("                COMPREHENSIVE GRAPH GENERATION EVALUATION")
    print("="*80)
    print(" EVALUATION PROTOCOL:")
    print("• Model: Conditional WGAN-GP Graph Generator")
    print("• Evaluation Strategy: Generated vs Test Set Comparison")
    print("• Novelty Assessment: Generated vs Training Set Comparison")
    print("• Metrics: MMD (distributional), Uniqueness, Novelty, Validity")
    print("="*80)
    
    # 1. Overall evaluation (using test set)
    overall_results = evaluate_model(generator, test_loader, train_loader, device, dataset_stats, num_classes, num_samples=300)
    
    # 2. Class-specific evaluation (using test set)
    print("\n" + "="*80)
    print("                    CLASS-SPECIFIC EVALUATION")
    print("="*80)
    print(" PURPOSE: Assess model performance per class to detect class-specific biases")
    print(" METHODOLOGY: Generate class-conditional samples and compare to class-specific test data")
    print("="*80)
    
    class_0_results = evaluate_class_specific(generator, test_loader, train_loader, device, dataset_stats, num_classes, target_class=0, num_samples=150)
    class_1_results = evaluate_class_specific(generator, test_loader, train_loader, device, dataset_stats, num_classes, target_class=1, num_samples=150)
    
    # 3. Visual inspection
    print("\n" + "="*80)
    print("                      VISUAL INSPECTION")
    print("="*80)
    print("  Generating sample graphs for qualitative assessment...")
    visualize_generated_graphs(generator, device, dataset_stats, num_samples=6)
    
    # 4. Save comprehensive results
    os.makedirs('./results', exist_ok=True)
    results_dir = os.path.dirname('./results')
    results_file = os.path.join(results_dir, "evaluation_results.txt")
    
    with open(results_file, 'w') as f:
        f.write("COMPREHENSIVE GRAPH GENERATION EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("EVALUATION METHODOLOGY:\n")
        f.write("- Model: Conditional WGAN-GP Graph Generator\n")
        f.write("- Ground Truth: Test set graphs (unseen during training)\n")
        f.write("- Novelty Baseline: Training set graphs\n")
        f.write("- Sample Sizes: 300 overall, 150 per class\n")
        f.write("- Features: 6D structural (nodes, edges, density, degrees, clustering)\n")
        f.write("- MMD Kernels: RBF (gamma=1.0) and Linear\n\n")
        
        f.write("METRIC DEFINITIONS:\n")
        f.write("- MMD: Maximum Mean Discrepancy (lower = better distribution match)\n")
        f.write("- Uniqueness: Fraction of unique generated graphs (higher = less mode collapse)\n")
        f.write("- Novelty: Fraction of generated graphs not in training set (higher = more creative)\n")
        f.write("- Validity: Fraction of structurally valid generated graphs (higher = better quality)\n\n")
        
        f.write("OVERALL RESULTS:\n")
        f.write(f"MMD DEGREE:     {overall_results['mmd_degree']:.6f}\n")
        f.write(f"MMD CLUSTERING:     {overall_results['mmd_clust']:.6f}\n")
        f.write(f"MMD LABELS:     {overall_results['mmd_label']:.6f}\n")
        f.write(f"MMD ORBIT:     {overall_results['mmd_orbit']:.6f}\n")
        f.write(f"Uniqueness:           {overall_results['uniqueness']:.4f} ({overall_results['uniqueness']*100:.1f}%)\n")
        f.write(f"Novelty:              {overall_results['novelty']:.4f} ({overall_results['novelty']*100:.1f}%)\n")
        f.write(f"Validity:             {overall_results['validity']:.4f} ({overall_results['validity']*100:.1f}%)\n\n")
        
        f.write("STRUCTURAL STATISTICS:\n")
        f.write(f"Average Nodes  - Test: {overall_results['test_stats']['num_nodes']:.2f}, Generated: {overall_results['generated_stats']['num_nodes']:.2f}\n")
        f.write(f"Average Edges  - Test: {overall_results['test_stats']['num_edges']:.2f}, Generated: {overall_results['generated_stats']['num_edges']:.2f}\n")
        f.write(f"Average Density - Test: {overall_results['test_stats']['density']:.4f}, Generated: {overall_results['generated_stats']['density']:.4f}\n\n")
        
        f.write("CLASS-SPECIFIC RESULTS:\n")
        f.write(f"Class 0 - MMD Degree: {class_0_results['mmd_degree']:.6f}, MMD Clustering: {class_0_results['mmd_clust']:.6f}, MMD Label: {class_0_results['mmd_label']:.6f}, MMD Orbit: {class_0_results['mmd_orbit']:.6f}, Uniqueness: {class_0_results['uniqueness']:.4f}, Novelty: {class_0_results['novelty']:.4f}, Validity: {class_0_results['validity']:.4f}\n")
        f.write(f"Class 1 - MMD Degree: {class_1_results['mmd_degree']:.6f}, MMD Clustering: {class_1_results['mmd_clust']:.6f}, MMD Label: {class_1_results['mmd_label']:.6f}, MMD Orbit: {class_1_results['mmd_orbit']:.6f}, Uniqueness: {class_1_results['uniqueness']:.4f}, Novelty: {class_1_results['novelty']:.4f}, Validity: {class_1_results['validity']:.4f}\n\n")
        
        f.write("PERFORMANCE ASSESSMENT:\n")
        overall_mmd_quality = "Excellent" if overall_results['mmd_clust'] < 0.01 else "Good" if overall_results['mmd_clust'] < 0.1 else "Poor"
        f.write(f"Overall Distributional Quality: {overall_mmd_quality}\n")
        f.write(f"Overall Diversity: {'High' if overall_results['uniqueness'] > 0.8 else 'Medium' if overall_results['uniqueness'] > 0.5 else 'Low'}\n")
        f.write(f"Overall Novelty: {'High' if overall_results['novelty'] > 0.7 else 'Medium' if overall_results['novelty'] > 0.4 else 'Low'}\n")
        f.write(f"Overall Validity: {'High' if overall_results['validity'] > 0.9 else 'Medium' if overall_results['validity'] > 0.7 else 'Low'}\n\n")
        
        f.write("NOTES:\n")
        f.write("- Evaluation uses TEST SET for distributional comparison (MMD)\n")
        f.write("- Novelty computed against TRAINING SET to detect memorization\n")
        f.write("- Lower MMD indicates better match to real data distribution\n")
        f.write("- Higher Uniqueness/Novelty/Validity scores are better\n")
    
    print(f"\n Detailed evaluation report saved to: {results_file}")
    
    print("\n" + "="*80)
    print("                   EVALUATION SUMMARY")
    print("="*80)
    overall_mmd_quality = "Excellent" if overall_results['mmd_clust'] < 0.01 else "Good" if overall_results['mmd_clust'] < 0.1 else "Poor"
    print(f" Diversity: {overall_results['uniqueness']*100:.1f}% unique graphs")
    print(f" Creativity: {overall_results['novelty']*100:.1f}% novel graphs")
    print(f" Quality: {overall_results['validity']*100:.1f}% valid graphs")
    print("="*80)
    print(" COMPREHENSIVE EVALUATION COMPLETE!")
    print("="*80)
    
    return {
        'overall': overall_results,
        'class_0': class_0_results,
        'class_1': class_1_results
    }


# --------------------------
#  Conditional Discriminator
# --------------------------
class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim, class_dim, hidden_dim, gnn_type="GCN"):
        super().__init__()
        self.class_embed_dim = hidden_dim  # embedding dim for classes
        self.class_embedding = nn.Embedding(class_dim, self.class_embed_dim)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim + self.class_embed_dim, 1)

    def forward(self, data, class_labels):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # normalize node features
        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)

        # GCN node embeddings
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Pool to graph-level embeddings
        x = global_mean_pool(x, batch)

        # Class embeddings
        label_embed = self.class_embedding(class_labels.to(x.device))
        x = torch.cat([x, label_embed], dim=1)

        return self.fc(x)


# --------------------------
#  Conditional Generator with Dynamic Node Count
# --------------------------
class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim, class_dim, hidden_dim, out_node_feat_dim, class_embed_dim=16):
        super().__init__()
        self.noise_dim = noise_dim
        self.class_dim = class_dim
        self.class_embedding = nn.Embedding(class_dim, class_embed_dim)
        self.input_dim = noise_dim + class_embed_dim
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_node_feat_dim)
        )

    def sample_node_count(self, class_label, dataset_stats):
        """Sample number of nodes based on dataset statistics for the given class"""
        if dataset_stats is None or class_label not in dataset_stats:
            return random.randint(10, 30)  # fallback
        
        stats = dataset_stats[class_label]
        # Sample from normal distribution with class-specific mean and std
        # Clamp to reasonable bounds
        mean = stats['mean']
        std = max(stats['std'], 1.0)  # avoid too small std
        sampled = int(np.random.normal(mean, std))
        
        # Clamp to dataset range with some buffer
        min_nodes = max(3, int(stats['min']))
        max_nodes = min(100, int(stats['max']) + 10)  # allow slightly larger
        
        return max(min_nodes, min(max_nodes, sampled))


    def forward(self, num_graphs=16, class_labels=None, dataset_stats=None, num_nodes=None):
        if class_labels is None:
            class_labels = torch.randint(0, self.class_dim, (num_graphs,), device=next(self.parameters()).device)
        
        label_embed = self.class_embedding(class_labels)  # [num_graphs, class_embed_dim]
        node_feats, edge_indices, batch, node_labels = [], [], [], []

        node_offset = 0

        for i in range(num_graphs):
            # Sample number of nodes for this graph based on class
            current_num_nodes = self.sample_node_count(class_labels[i].item(), dataset_stats) if num_nodes is None else num_nodes

            # Graph-level latent vector
            z_graph = torch.randn(1, self.noise_dim, device=label_embed.device).repeat(current_num_nodes, 1)
            label = label_embed[i].unsqueeze(0).repeat(current_num_nodes, 1)
            input = torch.cat([z_graph, label], dim=1)

            x = self.fc(input)

            # Random edges - scale probability with class and graph size
            edge_prob = 0.05 + 0.1 * (class_labels[i].item() == 1)  
            edge_prob = max(0.02, edge_prob * (20.0 / current_num_nodes))  

            edges = []
            for u in range(current_num_nodes):
                for v in range(u + 1, current_num_nodes):  
                    if random.random() < edge_prob:
                        edges.append([u + node_offset, v + node_offset])
                        edges.append([v + node_offset, u + node_offset])  

            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

            node_feats.append(x)
            edge_indices.append(edge_index)
            batch.append(torch.full((current_num_nodes,), i, dtype=torch.long, device=x.device))
            node_labels.append(torch.full((current_num_nodes,), class_labels[i], dtype=torch.long, device=x.device))  # add labels

            node_offset += current_num_nodes

        x = torch.cat(node_feats, dim=0)
        edge_index = torch.cat(edge_indices, dim=1) if edge_indices[0].numel() > 0 else torch.empty((2, 0), dtype=torch.long, device=x.device)
        batch = torch.cat(batch, dim=0)
        y = torch.cat(node_labels, dim=0)  # graph labels per node

        return Data(x=x, edge_index=edge_index, batch=batch, y=y)

# --------------------------
#  Gradient Penalty
# --------------------------
def compute_gradient_penalty(discriminator, real_data, fake_data, class_labels, device):
    min_nodes = min(real_data.x.size(0), fake_data.x.size(0))
    real_x, fake_x = real_data.x[:min_nodes], fake_data.x[:min_nodes]
    batch = real_data.batch[:min_nodes]

    alpha = torch.rand(batch.max().item()+1, 1, device=device)
    alpha_nodes = alpha[batch]

    interpolated_x = alpha_nodes * real_x + (1 - alpha_nodes) * fake_x
    interpolated_x.requires_grad_(True)

    edge_index = real_data.edge_index
    mask = (edge_index[0] < min_nodes) & (edge_index[1] < min_nodes)
    edge_index = edge_index[:, mask]

    interpolated_data = Data(x=interpolated_x, edge_index=edge_index, batch=batch).to(device)
    class_labels = class_labels[:batch.max().item()+1]

    d_interpolated = discriminator(interpolated_data, class_labels)

    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated_data.x,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grad_norm = gradients.norm(2, dim=1)
    grad_norm_per_graph = scatter(grad_norm**2, batch, dim=0, reduce='sum').sqrt()
    return ((grad_norm_per_graph - 1) ** 2).mean()


# --------------------------
#  Training Loop
# --------------------------
def train(generator, discriminator, train_loader, n_critic, lambda_gp, epochs, device, dataset_stats, num_classes):
    
    opt_g = torch.optim.Adam(generator.parameters(), lr=5e-5, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.0, 0.9))

    for epoch in range(1, epochs+1):
        generator.train()
        discriminator.train()
        d_losses, g_losses = [], []

        for batch in train_loader:
            batch = batch.to(device)
            batch_size = batch.y.size(0)
            real_labels = batch.y

            # --- Train Critic ---
            for _ in range(n_critic):
                fake_data = generator(num_graphs=batch_size, class_labels=real_labels, dataset_stats=dataset_stats)
                pred_real = discriminator(batch, real_labels)
                pred_fake = discriminator(fake_data.detach(), real_labels)
                gp = compute_gradient_penalty(discriminator, batch, fake_data, real_labels, device)

                d_loss = -torch.mean(pred_real) + torch.mean(pred_fake) + lambda_gp * gp
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                d_losses.append(d_loss.item())

            # --- Train Generator ---
            fake_data = generator(num_graphs=batch_size, class_labels=real_labels, dataset_stats=dataset_stats)
            pred_fake = discriminator(fake_data, real_labels)
            g_loss = -torch.mean(pred_fake)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            g_losses.append(g_loss.item())

        # Optional: log mean D(real) and D(fake) for stability check
        with torch.no_grad():
            real_vals = []
            fake_vals = []
            for batch in train_loader:
                batch = batch.to(device)
                real_vals.append(discriminator(batch, batch.y.to(device)))
                fake_batch = generator(num_graphs=batch.y.size(0), class_labels=batch.y.to(device), dataset_stats=dataset_stats).to(device)
                fake_vals.append(discriminator(fake_batch, batch.y.to(device)))
                if len(real_vals) >= 5:  # Limit for efficiency
                    break
            
            real_vals = torch.cat(real_vals)
            fake_vals = torch.cat(fake_vals)
            
        mean_d_real = real_vals.mean().item()
        mean_d_fake = fake_vals.mean().item()

        print(f"Epoch {epoch:02d}  D_loss: {sum(d_losses)/len(d_losses):.4f}  G_loss: {sum(g_losses)/len(g_losses):.4f}  "
              f"Mean D(real): {mean_d_real:.4f}  Mean D(fake): {mean_d_fake:.4f}")

    # Save models
    os.makedirs('./saved_models/ENZYMES_WGANGP', exist_ok=True)
    torch.save(generator.state_dict(), './saved_models/ENZYMES_WGANGP/generator.pt')
    torch.save(discriminator.state_dict(), './saved_models/ENZYMES_WGANGP/discriminator.pt')
    print("Models saved.")

# --------------------------
#  Eval
# --------------------------

def evaluate(generator, train_loader, test_loader, device, dataset_stats, num_classes):

    model_path = "./saved_models/ENZYMES_WGANGP/generator.pt"
    
     # Load model weights
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    print("Model loaded successfully!")
    
    results = run_complete_evaluation(generator, train_loader, test_loader, device, dataset_stats, num_classes)
    print("\nQuick Summary:")
    print(f"Overall MMD Degree: {results['overall']['mmd_degree']:.6f}")
    print(f"Overall MMD Clustering: {results['overall']['mmd_clust']:.6f}")
    print(f"Overall MMD Label: {results['overall']['mmd_label']:.6f}")
    print(f"Overall MMD Orbit: {results['overall']['mmd_orbit']:.6f}")

    print(f"Uniqueness: {results['overall']['uniqueness']:.4f}")
    print(f"Novelty: {results['overall']['novelty']:.4f}")
    print(f"Validity: {results['overall']['validity']:.4f}")
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TUDataset(root='./data/ENZYMES', name='ENZYMES').shuffle()
    
    # Analyze dataset statistics before splitting
    print("Analyzing dataset statistics...")
    dataset_stats = analyze_dataset_statistics(dataset)
    
    num_classes = dataset.num_classes
    
    train_dataset, test_dataset = dataset[:500], dataset[500:]
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    noise_dim = 16
    n_critic = 5
    lambda_gp = 10
    epochs = 30

    num_node_features = dataset.num_node_features
    print(f"Node features: {num_node_features}")
    num_classes = dataset.num_classes

    generator = ConditionalGenerator(noise_dim=noise_dim, class_dim=num_classes,
                                     hidden_dim=32, out_node_feat_dim=num_node_features).to(device)
    discriminator = ConditionalDiscriminator(input_dim=num_node_features,
                                             class_dim=num_classes, hidden_dim=64).to(device)
    
    print(f"\nDataset info:")
    print(f"Total graphs: {len(dataset)}")
    print(f"Training graphs: {len(train_dataset)}")
    print(f"Test graphs: {len(test_dataset)}")
    print(f"Node features: {num_node_features}")
    print(f"Classes: {num_classes}")
    
    train(generator, discriminator, train_loader, n_critic, lambda_gp, epochs, device, dataset_stats, num_classes)
    evaluate(generator, train_loader, test_loader, device, dataset_stats, num_classes)

# --------------------------
#  Usage Example
# --------------------------

if __name__ == "__main__":
    main()