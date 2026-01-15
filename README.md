# 🛰️ Land Use and Land Cover (LULC) Classification of Noord-Brabant, Netherlands

This project uses deep learning architectures to map land cover and cropland. There are two phases:

- **Phase 1:** Fine-tune a pretrained **Vision Transformer (ViT)** (pretrained on ImageNet-21k) on the EuroSAT RGB dataset. The model is then deployed to classify the Noord-Brabant Province in the Netherlands into 10 LULC classes.
- **Phase 2:** Mask out non-vegetative classes using the LULC results from Phase 1 and utilize the **Prithvi foundation model** for fine-grained crop classification. Loading.....

The project is structured modularly, powered by **Pipenv** for dependency management and reproducible **YAML** configurations.

## 📂 Project Structure

```

├── config.yaml           # Configuration for training and data parameters
├── data/                 # Directory for EuroSAT and local geospatial data
├── outputs/              # Stores trained weights (.pth), metric logs, and output GeoTIFFs
├── datafactory.py        # PyTorch Dataset wrappers and augmentation pipelines
├── engine.py             # Reusable training, validation, and optimization loops
├── inference.py          # Sentinel-2 image processing for inference and map generation
├── utils.py              # Helper functions for seeding, subsetting, and visualization
├── Phase_1A.ipynb        # ViT model training
├── Phase_1B.ipynb        # Earth Engine data sourcing and large-scale inference
├── Pipfile               # Reproducible environment with Pipenv
└── .gitignore            # Ensures large data and model files are not tracked by Git

```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone 
cd LULC-Brabant
````

### 2. Set up the Environment

```bash
# Make sure Pipenv is installed
pip install pipenv

# Create and activate virtual environment
set PIPENV_VENV_IN_PROJECT=1
pipenv install --dev
pipenv shell
```

### 3. Download the EuroSAT RGB Dataset

* Link: [https://github.com/phelber/EuroSAT](https://github.com/phelber/EuroSAT)
* Extract into the following structure:

```
data/
└── EuroSAT/
    └── 2750/
        ├── AnnualCrop/
        ├── Forest/
        └── ...
```

---

## ⚙️ Configuration

All settings can be found in `config.yaml`. Example:

```yaml
data_dir: "./data/EuroSAT/2750/"
batch_size: 16
num_epochs: 10
lr: 0.001
weight_decay: 0.05
percentage_per_class: 0.3
```

---

## 🧠 Model: Vision Transformer (ViT)

I used the pretrained `vit_b_16` from `torchvision.models`, and fine-tune it to classify 10 LULC categories.



## 📊 Outputs

After executing the notebooks, you'll find:

* `best_model.pth`: The ViT weights with the highest validation accuracy achieved during Phase 1A.
* `train_results.json`: A log of Loss and Accuracy metrics for both training and validation sets from Phase 1A
* `brabant_map`: LULC map of Noord-Brabant with 10 classes from Phase1B

---

## 🧑‍💻 Author

**Henry Osei**

---


## 📜 License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

```

