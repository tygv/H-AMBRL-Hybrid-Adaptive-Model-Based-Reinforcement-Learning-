
# Hybrid Adaptive Model-Based Reinforcement Learning (H-AMBRL)

This repository contains the implementation of **Hybrid Adaptive Model-Based Reinforcement Learning (H-AMBRL)** for optimizing fast-charging protocols in lithium-ion batteries. The model integrates **Gaussian Process Regression (GP), Neural Networks (NN), and Random Forest (RF)** to enhance the charging efficiency while ensuring battery longevity.

## 📂 Project Structure
H-AMBRL/
│-- datasets/                # Battery datasets (CSV files)
│-- scripts/                 
│   │-- hard_vs_trad_v3.py    # Hardware-based model implementation
│   │-- online_vs_trad_v3.py  # Online-based model implementation
│-- logs/                    
│   │-- hardware_log.txt      # Log file for hardware-based runs
│   │-- online_vs_trad_log.txt # Log file for online-based runs
│-- README.md                 # Project documentation
│-- .gitignore                 # Ignore unnecessary files


## 🚀 Features
- ✅ **Multi-Model Learning**: Integrates GP, NN, RF to improve fast-charging predictions.
- ✅ **Reinforcement Learning (RL)**: Implements PPO-based reinforcement learning.
- ✅ **Real & Simulated Data Support**: Uses both **hardware-based** and **online-based** datasets.
- ✅ **Data Processing & Visualization**: Generates **plots**, **learning curves**, and **feature importance**.

## ⚡ Dataset
The datasets contain battery parameters collected under different charging conditions:
- **Voltage (V)**
- **Current (A)**
- **State of Charge (SoC)**
- **Remaining Capacity (Ah)**
- **Cycle Index**
- **dV/dt**

To use the datasets:
1. Download them from the `datasets/` folder.
2. Ensure CSV files have the required columns.
3. Run the scripts to train and evaluate the models.

## 🔧 Setup Instructions
### **1. Clone the Repository**
```sh
git clone https://github.com/tygv/H-AMBRL.git
cd H-AMBRL
