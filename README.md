# Hybrid Adaptive Model-Based Reinforcement Learning (H-AMBRL)

This repository contains the implementation of **Hybrid Adaptive Model-Based Reinforcement Learning (H-AMBRL)** for optimizing fast-charging protocols in lithium-ion batteries. The model integrates **Gaussian Process Regression (GP), Neural Networks (NN), and Random Forest (RF)** to enhance charging efficiency while ensuring battery longevity.

## 📂 Project Structure
```
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
```

## 🚀 Features
- ✅ **Multi-Model Learning**: Integrates GP, NN, RF to improve fast-charging predictions.
- ✅ **Reinforcement Learning (RL)**: Implements PPO-based reinforcement learning.
- ✅ **Real & Simulated Data Support**: Uses both **hardware-based** and **online-based** datasets.
- ✅ **Data Processing & Visualization**: Generates **plots**, **learning curves**, and **feature importance**.

## ⚡ Dataset

### **1. Hardware-Based Dataset**
The hardware dataset consists of real-world battery charging data collected under different conditions.

**Features included:**
- **Voltage (V)**
- **Current (A)**
- **State of Charge (SoC)**
- **Remaining Capacity (Ah)**
- **Cycle Index**
- **dV/dt**

The dataset is stored in the `datasets/` folder and is used by `hard_vs_trad_v3.py`.

### **2. Online Dataset**
For additional real-world data, we use the publicly available dataset from **Matrian**:
🔗 **[Online Dataset](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204/batches/5c86bd64fa2ede00015ddbb3)**

#### **Channels Used:**
The model is trained using the following channels:
```
7, 9, 10, 11, 26, 31, 38, 39, 44, 46
```

---

## 🔧 Setup Instructions

### **1. Clone the Repository**
```sh
git clone https://github.com/tygv/H-AMBRL.git
cd H-AMBRL
```

### **2. Install Dependencies**
Ensure you have **Python 3.8+** installed, then run:
```sh
pip install -r requirements.txt
```

### **3. Run the Scripts**
To run the **hardware-based model**:
```sh
python scripts/hard_vs_trad_v3.py
```

To run the **online-based model**:
```sh
python scripts/online_vs_trad_v3.py
```

### **4. View Logs**
After execution, check the log files in the `logs/` folder.

---

## 📊 Results
After training, the scripts will generate:
- **Predicted vs Actual Plots**
- **Residual Analysis**
- **Feature Importance (Random Forest)**
- **Learning Curve (PPO Reinforcement Learning)**

---

## 🛠️ Hardware & Software Requirements
- **Python 3.8+**
- **Libraries**: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `gymnasium`, `stable-baselines3`
- **Hardware**: Tested on `Intel Core i7`, `16GB RAM`, `GPU (Optional)`

---

## 🤝 Contributing
If you’d like to contribute:
1. Fork this repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Added new feature"`).
4. Push to GitHub (`git push origin feature-branch`).
5. Submit a **Pull Request**.

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 📧 Contact
For questions, contact:
📩 **tr18000236@studentmail.biust.ac.bw**
