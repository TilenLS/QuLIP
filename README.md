# QuLIP: A Variational Quantum Encoder for Vision-Language Understanding

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-purple.svg)](https://pennylane.ai/)
[![Lambeq](https://img.shields.io/badge/Lambeq-QNLP-green.svg)](https://cqcl.github.io/lambeq/)

> **Official implementation of "Meaning Representations as Variational Quantum Circuits"**
> 
> *Tilen G. Limbäck-Stokin, Tanishka A. Birdavade, Kin Ian Lo, Mehrnoosh Sadrzadeh* > Quantum Learning Labs, University College London (UCL)

## 📖 Overview

Classical Vision-Language Models (VLMs) like CLIP rely heavily on unstructured sequences and $O(n^2)$ self-attention, ignoring the compositional nature of language and leading to massive parameter explosions. **QuLIP (Quantum Language-Image Pretraining)** addresses this by mapping syntactic rules—specifically Combinatory Categorial Grammar (CCG)—directly into **Variational Quantum Circuits (VQCs)**.

By treating grammatical compositions as quantum entangling operations and words as parameterized unitary rotations, QuLIP achieves **competitive multimodal alignment (e.g., 83.16% on SVO-Swap)** while utilizing **two orders of magnitude fewer parameters** (10k-100k) than classical baselines like OpenCLIP (63M).

![QuLIP Pipeline](assets/pipeline.png) 
*Figure 1: The QuLIP multimodal pipeline. Images are classically embedded and amplitude-encoded. Sentences are parsed via CCG and topologically mapped to VQCs to generate the text state. Alignment is computed via a quantum inner product.*

## 🧬 Repository Structure

The codebase is modularized to support CCG parsing, topological circuit compilation, scalable tensor contraction, and custom quantum loss landscapes.

* `data_processing.py`: Handles multimodal dataset ingestion (ARO, SVO-Swap), CLIP image extraction, and maps classical embeddings to PennyLane `AmplitudeEmbedding` states.
* `tree2einsum.py` & `grammar_ext.py`: The core NLP-to-Quantum compiler. Translates Bobcat CCG derivation trees into parameterized unitary ansätze (e.g., `Sim14Ansatz`, `IQPAnsatz`, `BrickworkAnsatz`).
* `model.py`: Contains the `EinsumModel` built on PyTorch and `cotengra`. Compiles VQCs into optimized `einsum` contraction paths for highly scalable, batched quantum simulation. Also contains the **QInfoNCE** loss variants.
* `util.py`: Implements fast, batched quantum gate operations (e.g., `BatchRz`, `BatchCRx`) and the Fubini-Study distance metrics.
* `visualisation.py`: A comprehensive suite for generating PCA-projected 3D loss landscapes, t-SNE alignments, and fidelity density distributions using `plotly` and `seaborn`.
* `example_training.ipynb`: A complete end-to-end training and evaluation loop leveraging `MLflow` for hyperparameter tracking.

## ⚙️ Core Methodologies

### 1. Syntax to Quantum Circuits
We replace classical grammatical function applications with native quantum operations. Words are parameterized by ansätze acting on the $|0\rangle$ state, and function applications (like subject-verb-object reductions) are achieved via Bell-basis measurements and post-selection to the $(|00\rangle+|11\rangle)/\sqrt{2}$ state.

![CCG to VQC Compilation](assets/circuit.png)
*Figure 2: The structural compilation of "Alice likes Bob". Inductive grammatical biases are explicitly encoded into the circuit topology.*

### 2. The QInfoNCE Objective & Fubini-Study Metric
Standard quantum fidelity creates sharp loss landscapes prone to barren plateaus. To align text and image states contrastively, we employ a smooth Fubini-Study similarity metric:

$$s(|\psi_{txt}\rangle, |\psi_{img}\rangle) = \arcsin(|\langle\psi_{txt}|\psi_{img}\rangle|)$$

Implemented in `model.py` as `QInfoNCE_cos`, this allows for highly stable gradient descent during multimodal alignment.

![Loss Landscape](assets/loss_landscape.png)
*Figure 3: 3D projection of the parameter loss landscape during training, mapped via PCA.*

### 3. 📊 Results on Compositional Benchmarks
QuLIP successfully bypasses the "bag-of-words" collapse seen in classical and unstructured quantum models.

Model,Parameters,SVO-Swap,ARO Attribution,ARO Relation
Quantum BoW,10K - 100K,50.00%,50.00%,50.00%
MicroCLIP,100K,68.42%,50.85%,51.05%
CLIP,63M,57.89%,61.00%,51.53%
QuLIP (CCG-VQC),~90K,83.16%,71.19%,57.33%

## 🚀 Quickstart

### Installation
Ensure you have Python 3.12+ installed. The required dependencies include `torch`, `lambeq`, `pennylane`, `cotengra`, and `mlflow`.

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/QuLIP.git](https://github.com/YOUR_USERNAME/QuLIP.git)
cd QuLIP

# Install dependencies (uv recommended for speed)
pip install uv
uv pip install lambeq pandas tqdm pennylane torch clip mlflow cotengra optuna
uv pip install git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)
```

## 📜 Citation

If you use this codebase or find our work on quantum compositional semantics helpful, please cite our paper:
```
@inproceedings{limbackstokin2026meaning,
  title={Meaning Representations as Variational Quantum Circuits},
  author={Limb\"{a}ck-Stokin, Tilen G. and Birdavade, Tanishka A. and Lo, Kin Ian and Sadrzadeh, Mehrnoosh},
  booktitle={LREC},
  year={2026},
  organization={Quantum Learning Labs, University College London}
}
```


## 📬 Contact
For questions or collaborations, please reach out:
* **Tilen G. Limbäck-Stokin:** tilen.limback-stokin.21@ucl.ac.uk
* **Lab:** [Quantum Learning Labs, UCL](https://www.ucl.ac.uk/engineering/computer-science/research/research-groups-and-centres/programming-principles-logic-and-verification-group/quantum-learning-labs)


