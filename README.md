# Generating Labeled Graphs Using Conditional Wasserstein GANs

# Abstract
Graph-structured data arises in many domains, from biological and chemical networks to social and knowledge graphs, where capturing both structural and class-specific patterns is critical. Generating realistic graphs conditioned on target class labels remains a challenging problem due to the discrete and irregular nature of graph topology. In this work, we propose a conditional Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) for labeled graph generation. Our framework integrates class information at both the generator and discriminator, enabling controllable synthesis of graphs with desired properties. The generator maps random noise vectors and class embeddings to node feature and adjacency representations, while the discriminator leverages a Graph Neural Network to jointly evaluate graph authenticity and class consistency. We evaluate the approach on benchmark graph datasets, demonstrating its ability to generate structurally coherent and class-consistent graphs. Experimental results show improved stability, highlighting the framework's potential for applications in synthetic dataset augmentation, controlled graph generation, and downstream tasks.

---

## 🧩 Method Summary
This paper focuses on conditional graph generation, where the model learns to generate synthetic graphs conditioned on class labels. This enables controllable and class-aware graph synthesis while preserving both structural and label-specific patterns.

We train a conditional generator G(z, y) that takes a random noise vector z ~ N(0, I) and a class label y as input and outputs a synthetic graph Ĝ.
The goal is for the generated graphs to match the real labeled graph distribution P_r, capturing the underlying topological and node-feature patterns associated with each class.

🔹 Generator
Combines latent noise and learned class embeddings.
Passes them through fully connected layers to produce node embeddings.
Edges are sampled using class-dependent probabilities that scale inversely with graph size.
The result is a synthetic, label-conditioned graph G = (X, E).

🔹 Discriminator (Critic)
Based on a Graph Convolutional Network (GCN) as in Kipf & Welling (2017).
Aggregates node embeddings using global mean pooling to obtain a graph-level representation.
Concatenates this with the class embedding and outputs a Wasserstein score D(G, y) — indicating how real and label-consistent a graph is.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/155be277-5627-4c8e-833f-f60007d31e8f" />


---

## ⚙️ Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/ava-12/labled_wasserstein_GANs.git
cd labled_wasserstein_GANs
pip install torch torch-geometric numpy networkx matplotlib
python WGAN_GP_Gen.py

