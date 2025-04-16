# Master Thesis: Angular Reconstruction of High Energy Neutrino Events in IceCube using Transformers

## 🧊 Overview

This repository contains the complete codebase for my Master's thesis, focusing on the development and evaluation of a transformer-based machine learning model for the angular reconstruction of high-energy neutrino events detected by the IceCube Neutrino Observatory.

**Motivation:**
Over the last century, investigating elementary particles has significantly advanced physics, culminating in the Standard Model. Yet, many questions remain, particularly at the intersection of particle physics and astrophysics. Understanding high-energy processes in the universe, like those in active galactic nuclei, requires detailed knowledge of particle interactions. High-energy astrophysical particles detected on Earth reveal powerful cosmic accelerators, but their nature and location remain largely unknown.

Neutrinos, the elusive "ghost particles," are crucial messengers. They interact weakly, traversing the cosmos largely undisturbed, unlike high-energy γ-rays or cosmic rays. This makes them ideal probes for astronomical phenomena. Multi-messenger astronomy and the search for high-energy neutrino point sources motivated the construction of the IceCube Neutrino Observatory. IceCube has detected astrophysical neutrinos, linking them to distant blazars, active galactic nuclei, and potentially tidal disruption events, opening a new window into the high-energy universe.

**The Challenge:**
Neutrino astronomy primarily targets muon neutrinos (νμ) due to their track-like signatures, which allow for better angular resolution – critical for identifying neutrino sources amidst background noise. Traditional likelihood-based methods achieve good resolution (≲ 1° at TeV, ≲ 0.2° at PeV) but can be computationally intensive. Machine learning, particularly Graph Neural Networks (GNNs), has shown promise for faster reconstruction but hasn't consistently outperformed traditional methods at higher energies. The 2023 IceCube Kaggle competition highlighted the potential of the transformer architecture.

**This Project:**
This thesis builds upon recent advancements, employing a novel **PMT-fication** approach (aggregating pulses at the photomultiplier tube level) to develop a transformer model for reconstructing muon neutrino tracks (100 GeV - 100 PeV). The research explores:
* The impact of event selection, model size, training set size, and input data representation (PMT-fication) on transformer performance.
* Fundamental aspects of transformer behavior on IceCube data.
* Comparison of the final transformer model's angular resolution against state-of-the-art likelihood methods (SplineMPE).

The goal is not only to present the findings but also to provide a foundation for future research by making the code accessible and reproducible.

## 📄 Thesis Document

The full thesis document, containing detailed theoretical background, methodology, analysis, results, and discussion, is available in the root directory of this repository:

* **[Your Thesis Title].pdf** (`[Link to Thesis PDF]`) - *Please replace `[Link to Thesis PDF]` with the actual link or filename if it's directly in the repo.*

Part I covers the theoretical background (Standard Model, Neutrino Physics, IceCube, Machine Learning, Transformers, Traditional Reconstruction). Part II details the specific methods, data, model architecture, training, results, and conclusions of this research.

## 📁 Repository Structure

The codebase is organized into three main directories, plus analysis notebooks and the thesis PDF:

.├── [Your Thesis Title].pdf       # The full thesis document├── data_preparation/             # Scripts for data cleaning and preparation│   ├── cosmic_ray_cleaning.py│   ├── event_selection.py│   ├── pmt_fication.py│   └── ...                       # Other relevant preparation scripts├── training_and_inference/       # Core scripts for the transformer model│   ├── model.py                  # Transformer model class definition│   ├── dataset.py                # Custom Dataset class for IceCube data│   ├── dataloader.py             # Dataloader implementation (using PMT-fication)│   ├── loss.py                   # Loss function(s) used for training│   ├── train.py                  # Script to train the model│   ├── inference.py              # Script to run inference and evaluate the model│   └── ...                       # Config files, utility scripts, etc.├── analysis/                     # Analysis notebooks and scripts│   └── analysis_plots.ipynb      # Jupyter notebook to generate figures from the thesis├── requirements.txt              # Python dependencies└── README.md                     # This file
* **`data_preparation/`**: Contains all scripts related to preparing the raw data for the model. This includes cosmic ray cleaning, event selection based on specific criteria, and the novel PMT-fication process.
* **`training_and_inference/`**: This is the core directory. It holds the Python code defining the transformer architecture, the custom dataset and dataloader logic handling the PMT-fied data, loss functions, and the main scripts for training the model (`train.py`) and performing angular reconstruction on new data (`inference.py`).
* **`analysis/`**: Includes Jupyter notebooks (`.ipynb`) or Python scripts used to generate the plots, figures, and statistical analyses presented in the thesis.

## 🚀 Getting Started

Follow these steps to set up the environment and run the code.

### Prerequisites

* Python (specify version, e.g., 3.9+)
* Required Python packages (see `requirements.txt`)
* Access to the relevant IceCube dataset(s) (See Thesis Part II, Section X.Y for details on data). *Specify if data needs to be downloaded separately or generated.*

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd [Your Repository Name]
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Code

1.  **Data Preparation:**
    * Place the necessary raw input data in the designated location (specify where, e.g., a `data/raw/` directory).
    * Run the scripts in the `data_preparation/` directory in the required order (e.g., cleaning -> selection -> PMT-fication). Refer to the thesis (Part II) or script comments for specific instructions.
    ```bash
    # Example commands (adjust paths and arguments as needed)
    python data_preparation/cosmic_ray_cleaning.py --input data/raw/ --output data/cleaned/
    python data_preparation/event_selection.py --input data/cleaned/ --output data/selected/
    python data_preparation/pmt_fication.py --input data/selected/ --output data/pmt-fied/
    ```
2.  **Training:**
    * Configure the training parameters (e.g., in `train.py` or a config file).
    * Run the training script, pointing it to the prepared data.
    ```bash
    python training_and_inference/train.py --data_path data/pmt-fied/ --model_output_path models/ --config configs/default_config.yaml
    ```
3.  **Inference & Evaluation:**
    * Use the trained model to perform angular reconstruction on a test dataset.
    ```bash
    python training_and_inference/inference.py --data_path data/pmt-fied-test/ --model_path models/final_transformer.pth --output_path results/
    ```
4.  **Analysis:**
    * Open and run the Jupyter notebook(s) in the `analysis/` directory to reproduce the plots and figures from the thesis.
    ```bash
    jupyter notebook analysis/analysis_plots.ipynb
    ```

*Note: Specific command-line arguments and configurations might vary. Please refer to the scripts and the thesis document for detailed usage.*

## 🔄 Reproducibility

This repository aims to ensure the reproducibility of the results presented in the thesis.
* The `requirements.txt` file lists the necessary package versions.
* The scripts in `data_preparation/` allow for recreating the exact data processing steps.
* The `training_and_inference/` scripts, along with saved model weights (if provided) or training configurations, enable retraining or re-evaluating the model.
* The `analysis/analysis_plots.ipynb` notebook uses the output from the inference step to generate the key figures, allowing for direct comparison with the thesis results.

Please consult Part II of the thesis for details on the specific datasets, simulation parameters, event selection criteria, and evaluation metrics used.

## ✨ Extending the Work

This research opens several avenues for future exploration:
* Investigating different transformer architectures or attention mechanisms.
* Applying the PMT-fication and transformer approach to other reconstruction tasks (e.g., energy reconstruction, particle identification).
* Exploring alternative pulse aggregation or data representation techniques.
* Training on larger or more diverse datasets.
* Optimizing the model for deployment in the IceCube real-time system.

Feel free to fork this repository and build upon this work. Contributions and suggestions are welcome!

## 📊 Visualizations

Key results and concepts are visualized in the thesis. Consider adding some key figures here for a quick overview:

* **Angular Resolution Comparison:** A plot showing the angular resolution (e.g., median angular error vs. energy) of the transformer model compared to SplineMPE.
    ```markdown
    ![Angular Resolution Plot](path/to/your/resolution_plot.png)
    *(Caption: Comparison of angular resolution between the transformer model and SplineMPE.)*
    ```
* **Model Architecture:** A diagram illustrating the transformer architecture used.
    ```markdown
    ![Model Architecture Diagram](path/to/your/architecture_diagram.png)
    *(Caption: Diagram of the transformer architecture employed in this work.)*
    ```
* **PMT-fication Example:** A visual representation of how raw pulses are aggregated into the PMT-fied format.
    ```markdown
    ![PMT-fication Visualization](path/to/your/pmt_fication_viz.png)
    *(Caption: Illustration of the PMT-fication process.)*
    ```

*(Replace the `path/to/your/...` placeholders with the actual paths to your image files within the repository or hosted URLs.)*

## 📜 Citation

If you use this code or build upon the methods presented in this thesis, please cite the work appropriately. *[Provide citation details here - e.g., link to thesis archive, DOI if available, or a suggested BibTeX entry].*

```bibtex
@mastersthesis{your_thesis_key,
  author       = {Your Name},
  title        = {Your Thesis Title},
  school       = {Your University},
  year         = {Year},
  url          = {[Link to Thesis or Repository]}
}
📝 LicenseThis project is licensed under the [Your Chosen License - e.g., MIT License
