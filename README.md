# DAI - Dynamic Aging Index

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Deep learning AI for unsupervised learning of aging trajectories from longitudinal biological data.

## 📋 Abstract

Aging arises from the dynamic instability in the organism's physiological state, and by discovering coordinate transformations that simplify these nonlinear dynamics, we can apply linear theory to predict, assess, and potentially modulate aging trajectories. Building on this concept, we developed an unsupervised artificial neural network using a variational autoencoder (VAE) framework, constrained by the Koopman operator. This guides the latent space to capture physiological dynamics through the stochastic evolution of a single variable, the **"Dynamic Aging Index" (DAI)**, which exhibited an exponential increase with chronological age, enabling prediction of remaining lifespan and aligning with observed late-life mortality deceleration. The autocorrelation performance serves as an indicator of the model's capacity to accurately predict future states in the dynamics of aging. Moreover, DAI was sensitive to disease progression and reflected multimorbidity risk. Extracted features highlighted functional chromosomal regions, providing further insights into the biological mechanisms underlying aging.

## 🎯 Key Features

- **🧬 Unsupervised Learning**: Discovers aging trajectories without labeled aging data
- **⏰ Temporal Prediction**: Forecasts biological aging states 3-10 years into the future
- **📊 Dynamic Aging Index**: Single scalar metric quantifying biological age
- **🔬 Multi-modal Architecture**: Combines VAE and Koopman operator theory
- **🏥 Clinical Applications**: Disease progression monitoring and multimorbidity risk assessment
- **📈 High-dimensional Processing**: Handles 332,909+ DNA methylation features
- **🎛️ Interpretable Results**: Extracts meaningful biological insights

## 🏗️ Architecture Overview

### Core Components

1. **Variational Autoencoder (VAE)**
   - Encodes high-dimensional biological data to 25-dimensional latent space
   - Implements residual blocks for improved gradient flow
   - Uses L1/L2 regularization and dropout for robust learning

2. **Koopman Operator**
   - Models temporal dynamics in latent space using linear transformations
   - Handles both oscillatory and exponential aging dynamics
   - Supports multi-step temporal predictions

3. **Dynamic Aging Index (DAI)**
   - Maps latent representations to interpretable scalar aging scores
   - Enables direct biological age assessment
   - Facilitates clinical decision-making

### Model Architecture
```
Input (332,909 features) 
    ↓
VAE Encoder (25D latent space)
    ↓
Koopman Operator (temporal evolution)
    ↓
DAI Projection (scalar aging index)
    ↓
Future State Prediction
```

## 📦 Installation

### Prerequisites

- **Python** >= 3.8
- **pip** >= 20.0
- **CUDA-compatible GPU** (recommended for training)
- **16+ GB RAM** (for large biological datasets)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/dynamic-aging-index.git
   cd dynamic-aging-index
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   ```

## 🚀 Quick Start

### 1. Prepare Your Data

Ensure your biological data follows this structure:
- **Cross-sectional data**: General biological features (332,909 features)
- **Present state data**: Current biological measurements
- **Future state data**: Target aging states for temporal modeling

### 2. Run the Model

```bash
# Start Jupyter server
jupyter notebook

# Open and run Dynamic_Aging_Index.ipynb
```

### 3. Training Configuration

The model will automatically:
- ✅ Normalize features using MinMaxScaler
- ✅ Split data into train/validation/test sets
- ✅ Configure GPU acceleration
- ✅ Train for 1000 epochs with checkpointing
- ✅ Save model weights for inference

### 4. Model Inference

```python
# Load trained model
model = load_model_from_checkpoint('path/to/checkpoint')

# Predict DAI score
dai_score = model.predict(biological_data)

# Forecast future aging states
future_states = model.predict_future(present_data, time_horizon=3)
```

## 📊 Dataset Requirements

### Data Sources

This study is based on data from:
- **dbGAP**: Framingham Heart Study, Women's Health Initiative, Normative Aging Study
- **Gene Expression Omnibus (GEO)**: Gene expression datasets

### Access Instructions

1. **dbGAP Access**: Requires authorized access following dbGAP rules
2. **GEO Access**: Download datasets using GEO accession IDs
3. **Data Format**: Ensure data is in parquet format for efficient loading

### Expected Data Format

```python
# Cross-sectional data structure
cross_sectional_data = {
    'features': (n_samples, 332909),  # Biological measurements
    'age': (n_samples, 1),            # Chronological age
    'dai_target': (n_samples, 1)      # Target DAI values
}

# Temporal data structure
temporal_data = {
    'present_states': (n_samples, 332909),  # Current measurements
    'future_states': (n_samples, 332909),   # Future measurements
    'time_intervals': (n_samples, 1)        # Time differences
}
```

## 🔧 Configuration

### Model Parameters

```python
# Key model parameters
MODEL_CONFIG = {
    'input_shape': (332909,),      # Biological feature dimension
    'latent_dim': 25,              # Latent space dimension
    'batch_size': 54,              # Training batch size
    'learning_rate': 0.0001,       # Initial learning rate
    'epochs': 1000,                # Training epochs
    'num_complex_pairs': 5,        # Koopman complex eigenvalues
    'num_real': 15,                # Koopman real eigenvalues
    'time_step': 3,                # Prediction time horizon (years)
}
```

### Loss Function Weights

```python
LOSS_WEIGHTS = {
    'reconstruction_loss': 10.0,   # Data reconstruction accuracy
    'kl_loss': 1.0,               # Latent space regularization
    'linear_dynamics_loss': 100.0, # Temporal prediction accuracy
    'future_state_loss': 100.0,   # DAI prediction accuracy
    'l_inf_loss': 1.0,            # Maximum error control
}
```

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation


## 🙏 Acknowledgments

- Framingham Heart Study participants and investigators
- Women's Health Initiative research group
- Normative Aging Study contributors
- Gene Expression Omnibus database
- TensorFlow and Keras development teams

---

**Note**: This repository contains notebooks to reproduce the results from our study. For questions about data access or model implementation, please open an issue or contact the maintainers.
