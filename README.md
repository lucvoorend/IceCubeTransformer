# Master Thesis: Angular Reconstruction of High Energy Neutrino Events in IceCube using Machine Learning

**Author**: Luc Voorend  
**Thesis Date**: May 20, 2025  
**Code base language**: Python 🐍
## 🧊 Overview

Welcome! This repository hosts the complete codebase for my Master's thesis. The core focus is developing and evaluating a **transformer-based machine learning model** for reconstructing the arrival direction (angular reconstruction) of high-energy neutrino events detected by the **IceCube Neutrino Observatory** at the South Pole.

---

### 🌌 Motivation: Unveiling the High-Energy Universe

Physics has made incredible strides with the **Standard Model** ⚛️, but the universe still holds many mysteries, especially where particle physics meets astrophysics. Understanding extreme cosmic events requires deciphering high-energy particle interactions. While we detect particles from powerful cosmic accelerators, their origins remain largely unknown.

Enter the **neutrino** 👻 – the elusive "*ghost particle*". With almost no mass and no charge, neutrinos travel billions of light-years unimpeded, carrying direct information from their sources. Unlike γ-rays or cosmic rays, they aren't easily absorbed or deflected. This makes them perfect messengers for multi-messenger astronomy. The **IceCube Observatory** was built precisely for this, aiming to map the high-energy neutrino sky and has already linked neutrinos to distant cosmic sources.

---

### 🎯 The Challenge: Pinpointing Neutrino Sources

Identifying specific neutrino sources requires *excellent angular resolution*. IceCube primarily uses muon neutrinos ($\mu_\nu$) for their track-like signatures.
* **Traditional Methods:** Likelihood-based techniques achieve good resolution (e.g., ≲ 1° at TeV) but are computationally *slow*.
* **Machine Learning:** GNNs showed promise for speed but haven't consistently beaten traditional methods at high energies (> GeV). The 2023 IceCube Kaggle competition put the spotlight on the potential of **transformers**.

---

### ✨ This Project: Transformers & PMT-fication

This thesis leverages these recent advancements, introducing a novel approach called **PMT-fication** (aggregating detector pulses at the Photomultiplier Tube level) to optimize data for a **transformer model**. The key goals were:

* 🧠 Develop a transformer capable of reconstructing muon neutrino tracks (100 GeV - 100 PeV).
* 🔬 Investigate how factors like *event selection*, *model size*, *training data size*, and *input representation* (PMT-fication) affect performance.
* 📈 Compare the final model's angular resolution against the state-of-the-art **SplineMPE** likelihood method.

Ultimately, this repository aims to share these findings and provide a **reproducible and extendable** codebase for future research in neutrino astronomy.

---

## 📄 Thesis Document

The full thesis document, containing detailed theoretical background, methodology, analysis, results, and discussion, is available in the root directory of this repository:

* **[Angular_reconstruction_of_high_energy_neutrinos_using_machine_learning_Luc_Voorend].pdf** (`[Link to Thesis PDF]`) 

Part I covers the theoretical background (Standard Model, Neutrino Physics, IceCube, Machine Learning, Transformers, Traditional Reconstruction). Part II details the specific methods, data, model architecture, training, results, and conclusions of this research.

## 📁 Repository Structure

The codebase is organized into three main directories, plus the thesis PDF and requirements:

├── [Angular_reconstruction_of_high_energy_neutrinos_using_machine_learning_Luc_Voorend].pdf # The full thesis document  
├── data_preparation/ # Scripts for data cleaning and preparation  
│ ├── cosmic_ray_cleaning.py  
│ ├── event_selection.py  
│ ├── pmt_fication.py  
│ └── ... # Other relevant preparation scripts  
├── training_and_inference/ # Core scripts for the transformer model 
│ ├── src/
│ │ ├── model.py # Transformer model class definition  
│ │ ├── dataset.py # Custom Dataset class for IceCube data  
│ │ ├── dataloader.py # Dataloader implementation (using PMT-fication)  
│ │ ├── loss.py # Loss function(s) used for training  
│ │ ├── utils.py # Assertion functions for config file  
│ ├── train.py # Script to train the model  
│ ├── inference.py # Script to run inference and evaluate the model  
│ └── config.yaml # Config file controlling settings for training and inference 
├── analysis/ # Analysis notebooks and scripts  
│ └── analysis.ipynb # Jupyter notebook to generate figures from the thesis  
├── requirements.txt # Python dependencies  
└── README.md # This file  

* **`data_preparation/`**: Contains all scripts related to preparing the raw data for the model. This includes cosmic ray cleaning, event selection based on specific criteria, and the novel PMT-fication process.
* **`training_and_inference/`**: This is the core directory. It holds the Python code defining the transformer architecture, the custom dataset and dataloader logic handling the PMT-fied data, loss functions, and the main scripts for training the model (`train.py`) and performing angular reconstruction on new data (`inference.py`).
* **`analysis/`**: Includes Jupyter notebooks (`.ipynb`) to generate the plots, figures, and statistical analyses presented in the thesis.

## 🚀 Getting Started

Follow these steps to set up the environment and run the code.

### Prerequisites

* Python (version 3.9+)
* Required Python packages (see `requirements.txt`)
* Access to the relevant IceCube dataset(s) (See Thesis Part II, Section 6.1 for details on data) 

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
    * Configure the training parameters in `config.json`.
    * Run the training script `train.py`
    ```bash
    # For no hang up run on a Unix/Linux system, use:
    nohup python -u training_and_inference/train.py > logs/output.log &
    ```
3.  **Inference & Evaluation:**
    * Use the trained model to perform angular reconstruction on a test dataset.
    ```bash
    python training_and_inference/inference.py --data_path data/pmt-fied-test/ --model_path models/final_transformer.pth --output_path results/
    ```
4.  **Analysis:**
    * Open and run the Jupyter notebook(s) in the `analysis/` directory to reproduce the plots and figures from the thesis.
    ```bash
    jupyter notebook analysis/analysis.ipynb
    ```

*Note: Specific command-line arguments and configurations might vary. Please refer to the scripts and the thesis document for detailed usage.*

## 🔄 Reproducibility

This repository aims to ensure the reproducibility of the results presented in the thesis.
* The `requirements.txt` file lists the necessary package versions.
* The scripts in `data_preparation/` allow for recreating the exact data processing steps.
* The `training_and_inference/` scripts, along with the training configurations, enable retraining or re-evaluating the model.
* The `analysis/analysis_plots.ipynb` notebook uses the output from the inference step to generate the key figures, allowing for direct comparison with the thesis results.

Please consult Part II of the thesis for details on the specific datasets, simulation parameters, event selection criteria, and evaluation metrics used.

## ✨ Extending the Work

This research opens several avenues for future exploration:
* Investigating different transformer architectures or attention mechanisms.
* Applying the PMT-fication and transformer approach to other reconstruction tasks (e.g., energy reconstruction, particle identification).
* Exploring alternative pulse aggregation or data representation techniques.
* Training on larger or more diverse datasets.
* Extending from Monte Carlo to IceCube real data events.

Feel free to fork this repository and build upon this work. Contributions and suggestions are welcome!

## 🙏 Acknowledgements

* The code related to PMT-fication and event selection was written by [Cyan Jo](https://github.com/KUcyans).
* The CR_cleaning code has been written by [Johann Nikolaides](https://github.com/jn1707).
* Special thanks to [Troels Petersen](https://github.com/troelspetersen), [Inar Timiryasov](https://github.com/timinar) and [Jean-Loup Tastet](https://github.com/JLTastet) for supervising the project.

## 📜 Citation

If you use this code or build upon the methods presented in this thesis, please cite the work appropriately.

```bibtex
@mastersthesis{Voorend_2025,
  author       = {Luc Voorend},
  title        = {Angular reconstruction of high energy neutrinos using machine learning},
  school       = {University of Copenhagen},
  year         = {2025},
  url          = {https://github.com/lucvoorend/IceCube_master_thesis}
}
```

## 📝 License

This project is licensed under the [Your Chosen License - e.g., MIT License
