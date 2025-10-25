# Labeled_Wasserstein_GANs

# Abstract
Graph-structured data arises in many domains, from biological and chemical networks to social and knowledge graphs, where capturing both structural and class-specific patterns is critical. Generating realistic graphs conditioned on target class labels remains a challenging problem due to the discrete and irregular nature of graph topology. In this work, we propose a conditional Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) for labeled graph generation. Our framework integrates class information at both the generator and discriminator, enabling controllable synthesis of graphs with desired properties. The generator maps random noise vectors and class embeddings to node feature and adjacency representations, while the discriminator leverages a Graph Neural Network to jointly evaluate graph authenticity and class consistency. We evaluate the approach on benchmark graph datasets, demonstrating its ability to generate structurally coherent and class-consistent graphs. Experimental results show improved stability, highlighting the framework's potential for applications in synthetic dataset augmentation, controlled graph generation, and downstream tasks.

---

## 🧠 Overview
This project implements the models and experiments described in **"Generating Labeled Graphs Using Conditional Wasserstein GANs"**.  

---

## 🧩 Method Summary
Our goal is to generate synthetic graphs conditioned on class labels, enabling controllable generation of graph-structured data while preserving both structural and class-specific patterns.
Formally, we train a conditional generator
which maps a latent noise vector \\( z \sim \mathcal{N}(0,I) \\) and a class label \\( y \in \mathcal{Y} \\) to a synthetic graph \\( \hat{G} \\). The generated graphs are optimized to match the distribution of real labeled graphs \\( \mathbb{P}_r \\), allowing the model to capture both **topological** and **class-dependent** characteristics. --- ### 🔹 Generator The generator concatenates latent noise and learned class embeddings, then passes them through fully connected layers to produce **node embeddings**. Edges are sampled using **class-dependent probabilities** that scale inversely with graph size — each possible edge \\( (u,v) \\) exists with a probability determined by the class label and node count. This yields a synthetic, class-conditioned graph \\( G = (X, E) \\). --- ### 🔹 Discriminator (Critic) The discriminator uses a **GCN-based architecture** ([Kipf & Welling, 2017](https://arxiv.org/abs/1609.02907)) to aggregate node embeddings via **global mean pooling**. It concatenates this representation with the class embedding and outputs a **Wasserstein score** \\( D_\phi(G, y) \\), quantifying how close the generated graph is to the real distribution while enforcing label alignment. --- ### ⚙️ Training (WGAN-GP Framework) Training follows the **Wasserstein GAN with Gradient Penalty (WGAN-GP)** paradigm. **Critic loss:** ```math \mathcal{L}_D = -\mathbb{E}_{G \sim \mathbb{P}_r}[D_\phi(G,y)] + \mathbb{E}_{\hat{G} \sim \mathbb{P}_g}[D_\phi(\hat{G},y)] + \lambda \mathcal{L}_{gp} ``` **Generator loss:** ```math \mathcal{L}_G = - \mathbb{E}_{\hat{G} \sim \mathbb{P}_g}[D_\phi(\hat{G},y)] ``` The gradient penalty term \\( \mathcal{L}_{gp} \\) enforces the Lipschitz constraint. The critic is updated multiple times per generator step to ensure stable convergence. --- ### ✅ Key Advantage By integrating **class information** into both the generator and discriminator, this framework ensures that the generated graphs exhibit **class-specific structures and features**, overcoming the limitations of traditional (unconditional) graph generation models. --- Would you like me to make it **GitHub-renderable with math** (using MathJax via HTML `<script>` tag) so the equations display properly on GitHub pages too? By default, GitHub Markdown doesn’t render LaTeX unless you use a workaround.

Example:
> This work introduces a modified GraphGAN that replaces BFS-tree sampling with noise-based generation.  
> It produces realistic node embeddings and synthetic edges for protein-protein interaction graphs.

---

## ⚙️ Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt

