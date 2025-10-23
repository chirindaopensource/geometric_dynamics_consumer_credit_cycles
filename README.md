# `README.md` 

# Geometric Dynamics of Consumer Credit Cycles: A Complete Implementation

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.15892-b31b1b.svg)](https://arxiv.org/abs/2510.15892)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/geometric_dynamics_consumer_credit_cycles)
[![Discipline](https://img.shields.io/badge/Discipline-Econometrics%20%7C%20ML%20%7C%20Finance-00529B)](https://github.com/chirindaopensource/geometric_dynamics_consumer_credit_cycles)
[![Data Sources](https://img.shields.io/badge/Data-FRED-lightgrey)](https://fred.stlouisfed.org/)
[![Core Method](https://img.shields.io/badge/Method-Geometric%20Algebra%20%7C%20Linear%20Attention-orange)](https://github.com/chirindaopensource/geometric_dynamics_consumer_credit_cycles)
[![Analysis](https://img.shields.io/badge/Analysis-Regime%20Detection%20%7C%20Interpretability-red)](https://github.com/chirindaopensource/geometric_dynamics_consumer_credit_cycles)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-%2311557c.svg?style=flat&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/seaborn-%233776ab.svg?style=flat&logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)
[![PyYAML](https://img.shields.io/badge/PyYAML-gray?style=flat)](https://pyyaml.org/)
[![tqdm](https://img.shields.io/badge/tqdm-ff69b4?style=flat)](https://github.com/tqdm/tqdm)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
---

**Repository:** `https://github.com/chirindaopensource/geometric_dynamics_consumer_credit_cycles`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"Geometric Dynamics of Consumer Credit Cycles: A Multivector-based Linear-Attention Framework for Explanatory Economic Analysis"** by:

*   Agus Sudjianto
*   Sandi Setiawan

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, auditable, and extensible pipeline that executes the entire research workflow: from rigorous data validation and preprocessing to model training, post-hoc attribution analysis, and the generation of all final diagnostic reports.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `execute_geometric_credit_cycle_research`](#key-callable-execute_geometric_credit_cycle_research)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the methodologies presented in the 2025 paper "Geometric Dynamics of Consumer Credit Cycles." The core of this repository is the iPython Notebook `geometric_dynamics_consumer_credit_cycles_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings, from initial data validation to the final generation of all analytical tables and figures.

The paper introduces a novel framework using Geometric Algebra and Linear Attention to move beyond traditional correlation-based analysis of economic cycles. This codebase operationalizes the framework, allowing users to:
-   Rigorously validate and manage the entire experimental configuration via a single `config.yaml` file.
-   Process raw quarterly macroeconomic data through a causally pure pipeline, including growth transformations and rolling-window standardization.
-   Construct multivector embeddings that explicitly model the rotational (feedback) dynamics between variables.
-   Train the Linear Attention model using the paper's specified chronological, single-step update algorithm.
-   Generate a full suite of diagnostic and interpretability outputs, including temporal attribution, geometric component attribution, and PCA-based regime trajectory plots.
-   Conduct systematic robustness analysis through automated hyperparameter sweeps.

## Theoretical Background

The implemented methods are grounded in Geometric (Clifford) Algebra, deep learning, and econometric time series analysis.

**1. Geometric Algebra (GA) Embedding:**
The core innovation is representing the economic state not as a simple vector, but as a **multivector** in a Clifford Algebra. The geometric product of two vectors `a` and `b` decomposes their relationship into a scalar (projective) part and a bivector (rotational) part:
$$
ab = a \cdot b + a \wedge b
$$
This project implements the multivector embedding from Equation (3) of the paper, which includes scalar, vector, and bivector components. The bivector terms, such as `(x_{u,t} - x_{c,t})(e_u \wedge e_c)`, are designed to activate when variables diverge, capturing the "tension" that drives feedback spirals.

**2. Linear Attention Mechanism:**
The model uses Linear Attention to identify relevant historical precedents. The attended context vector `O_t` is computed as a weighted average of past information, where the weights are determined by the geometric similarity between the current state (query `Q_t`) and historical states (keys `K_τ`). The key equations implemented are (8), (9), and (10):
$$
S_t = \sum_{\tau \in \mathcal{W}_t} K_\tau V_\tau^\top, \quad Z_t = \sum_{\tau \in \mathcal{W}_t} K_\tau
$$
$$
O_t = \frac{Q_t^\top S_t}{Q_t^\top Z_t + \varepsilon}
$$
The similarity `Q_t^T K_τ` is a multivector-aware dot product, allowing the model to match not just variable levels but the underlying geometric interaction patterns.

## Features

The provided iPython Notebook (`geometric_dynamics_consumer_credit_cycles_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Task Architecture:** The entire pipeline is broken down into 23 distinct, modular tasks, each with its own orchestrator function, ensuring clarity, testability, and rigor.
-   **Configuration-Driven Design:** All study parameters are managed in an external `config.yaml` file, allowing for easy customization and replication without code changes.
-   **Causally Pure Data Pipeline:** Implements professional-grade time series preprocessing, including causally correct rolling-window operations and transformations, with a `valid_mask` system to prevent any look-ahead bias.
-   **High-Fidelity Model Implementation:** Includes a complete, from-scratch implementation of the multivector embedding, the custom shifted Leaky ReLU, and the chronological `batch_size=1` training loop as specified in the paper.
-   **Comprehensive Interpretability Suite:** Provides functions to generate all key analytical outputs from the paper, including temporal attention heatmaps, geometric occlusion attribution, component magnitude plots, and PCA trajectory analysis.
-   **Automated Robustness Analysis:** Includes a top-level function to automatically conduct hyperparameter sweeps, running the entire pipeline for each configuration and compiling the results.
-   **Automated Reporting and Archival:** Concludes by automatically generating all publication-ready plots, summary tables, and a complete, timestamped archive of all data, parameters, results, and environment metadata for perfect reproducibility.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Validation (Tasks 1-2):** Ingests and rigorously validates the `config.yaml` and the raw `pd.DataFrame` against a strict schema.
2.  **Data Preprocessing (Tasks 3-6):** Cleanses the data, applies growth transformations, performs rolling-window standardization, and constructs the final `(T, 11)` multivector embedding matrix.
3.  **Model Setup (Tasks 7-8):** Initializes all learnable parameters with best-practice schemes (Kaiming/Xavier) and defines the custom activation function.
4.  **Training (Tasks 9-16):** Implements the full forward pass (QKV projections, attention statistics, context vector, MLP head), computes the prediction and regularization losses, and executes the chronological, per-time-step training loop to produce the final trained parameters.
5.  **Post-Hoc Analysis (Tasks 17-20):** Uses the trained model to compute temporal attributions, geometric attributions, component magnitudes, and the PCA trajectory of the system's state.
6.  **Master Orchestration (Tasks 21-23):** Provides top-level functions to run the entire pipeline, conduct robustness sweeps, and generate all final deliverables.

## Core Components (Notebook Structure)

The `geometric_dynamics_consumer_credit_cycles_draft.ipynb` notebook is structured as a logical pipeline with modular orchestrator functions for each of the major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: `execute_geometric_credit_cycle_research`

The project is designed around a single, top-level user-facing interface function:

-   **`execute_geometric_credit_cycle_research`:** This master orchestrator function, located in the final section of the notebook, runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project, from data validation to the final report generation and archival.

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `torch`, `matplotlib`, `seaborn`, `pyyaml`, `tqdm`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/geometric_dynamics_consumer_credit_cycles.git
    cd geometric_dynamics_consumer_credit_cycles
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy torch matplotlib seaborn pyyaml tqdm
    ```

## Input Data Structure

The pipeline requires a `pandas.DataFrame` with a specific schema, as generated in the "Usage" example. All other parameters are controlled by the `config.yaml` file.

## Usage

The `geometric_dynamics_consumer_credit_cycles_draft.ipynb` notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell of the notebook, which demonstrates how to use the top-level `execute_geometric_credit_cycle_research` orchestrator:

```python
# Final cell of the notebook

# This block serves as the main entry point for the entire project.
if __name__ == '__main__':
    # --- 1. Generate/Load Inputs ---
    # A synthetic data generator is included in the notebook for demonstration.
    # In a real use case, you would load your data here.
    # consolidated_df_raw = pd.read_csv(...)
    
    # Load the model configuration from the YAML file.
    with open('config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)
        
    # Define the hyperparameter grid for robustness analysis.
    hyperparameter_grid = {
        'hidden_dimension_dh': [32, 64],
        'learning_rate_eta': [1e-4, 5e-5]
    }
    
    # --- 2. Execute Pipeline ---
    # Define the top-level directory for all outputs.
    RESULTS_DIRECTORY = "research_output"

    # Execute the entire research study.
    final_results = execute_geometric_credit_cycle_research(
        consolidated_df_raw=consolidated_df_raw, # Assumes this df is generated/loaded
        model_config=model_config,
        hyperparameter_grid=hyperparameter_grid,
        save_dir=RESULTS_DIRECTORY,
        base_random_seed=42,
        run_robustness_analysis=True,
        show_plots=True
    )
```

## Output Structure

The pipeline creates a `save_dir` with a highly structured set of outputs. A unique timestamped subdirectory is created for the primary run (e.g., `analysis_run_20251027_103000/`), containing:
-   `historical_fit.png` and `diagnostic_dashboard.png`: Publication-quality plots.
-   `regime_summary_table.csv`: The data-driven summary of economic regimes.
-   `full_results.pkl`: A complete archive of the primary run's results.
-   `trained_parameters.pth`: The final PyTorch model parameters.
-   `environment.json`: A record of the computational environment.

If robustness analysis is run, a top-level file `robustness_analysis_full_results.pkl` is also saved, containing the results from every run in the hyperparameter sweep.

## Project Structure

```
geometric_dynamics_consumer_credit_cycles/
│
├── geometric_dynamics_consumer_credit_cycles_draft.ipynb # Main implementation notebook
├── config.yaml                                           # Master configuration file
├── requirements.txt                                      # Python package dependencies
│
├── research_output/                                      # Example output directory
│   ├── analysis_run_20251027_103000/
│   │   ├── historical_fit.png
│   │   ├── diagnostic_dashboard.png
│   │   ├── regime_summary_table.csv
│   │   ├── full_results.pkl
│   │   ├── trained_parameters.pth
│   │   └── environment.json
│   │
│   └── robustness_analysis_full_results.pkl
│
├── LICENSE                                               # MIT Project License File
└── README.md                                             # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can easily modify all study parameters, including lookback horizons, model dimensions, activation function parameters, and regularization strengths, without altering the core Python code.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **Alternative Geometries:** Exploring other geometric algebras (e.g., Projective Geometric Algebra) or differential geometry frameworks (e.g., Riemannian manifolds) to model economic state spaces.
-   **GPU Acceleration:** While the current implementation is efficient, the chronological training loop could be further optimized or parallelized for GPUs for very large datasets or extensive hyperparameter searches.
-   **Alternative Attention Mechanisms:** Integrating other efficient attention mechanisms (e.g., Performers, Transformers-are-RNNs) and comparing their diagnostic outputs.
-   **Formal Backtesting:** Extending the framework to a formal out-of-sample forecasting or trading strategy backtest to quantify the economic value of the geometric signals.

## License

This project is licensed under the MIT License.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{sudjianto2025geometric,
  title   = {Geometric Dynamics of Consumer Credit Cycles: A Multivector-based Linear-Attention Framework for Explanatory Economic Analysis},
  author  = {Sudjianto, Agus and Setiawan, Sandi},
  journal = {arXiv preprint arXiv:2510.15892},
  year    = {2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Professional-Grade Implementation of the Geometric Dynamics Framework.
GitHub repository: https://github.com/chirindaopensource/geometric_dynamics_consumer_credit_cycles
```

## Acknowledgments

-   Credit to **Agus Sudjianto and Sandi Setiawan** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **PyTorch, Pandas, NumPy, Matplotlib, Seaborn, and Jupyter**.

--

*This README was generated based on the structure and content of the `geometric_dynamics_consumer_credit_cycles_draft.ipynb` notebook and follows best practices for research software documentation.*
