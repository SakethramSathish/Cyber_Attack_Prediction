# 🔐 Cyber Attack Prediction System

An advanced machine learning and deep learning solution for real-time network security threat detection. This system leverages ensemble learning, neural networks, and sophisticated data augmentation techniques to accurately identify and classify cyber attacks in network traffic.

## 🎯 Overview

The Cyber Attack Prediction System combines multiple state-of-the-art machine learning and deep learning approaches to detect malicious network activity with high accuracy. By utilizing ensemble methods, class balancing techniques, and advanced data augmentation, the system achieves superior performance in cybersecurity threat detection.

### Key Features

- **🤖 Machine Learning**: Random Forest Classifier with balanced class weights for robust baseline predictions
- **🧠 Deep Learning**: Neural networks with batch normalization and dropout for advanced pattern recognition
- **🎨 Data Augmentation**: SMOTE + Tomek Links hybrid sampling for optimal class balance
- **📊 Ensemble Voting**: Three-model consensus approach for high-confidence predictions
- **📈 Comprehensive Analytics**: Real-time performance metrics and detailed visualizations
- **🎯 Interactive UI**: User-friendly Streamlit interface for exploration and prediction

## 📋 Dataset Information

**Cybersecurity Intrusion Dataset**
- **Total Samples**: 1,536 network sessions
- **Features**: 10 network and behavioral features
- **Target Variable**: `attack_detected` (0 = Normal, 1 = Attack)
- **Feature Types**: Network metrics, session attributes, encryption protocols, browser information

### Feature Categories

- **Network Features**: Packet size, protocol type, session duration
- **Behavioral Features**: Login attempts, connection patterns
- **Security Features**: Encryption type, encryption status
- **Client Features**: Browser type

## 🛠️ Technology Stack

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **streamlit** | 1.28.1 | Web application framework |
| **pandas** | 2.1.3 | Data manipulation and analysis |
| **numpy** | 1.24.3 | Numerical computing |

### Machine Learning

| Library | Version | Purpose |
|---------|---------|---------|
| **scikit-learn** | 1.3.2 | ML algorithms and preprocessing |
| **imbalanced-learn** | 0.11.0 | Data balancing (SMOTE, Tomek Links) |
| **xgboost** | 2.0.3 | Gradient boosting classifier |
| **lightgbm** | 4.0.0 | Lightweight gradient boosting |

### Deep Learning

| Library | Version | Purpose |
|---------|---------|---------|
| **tensorflow** | 2.14.0 | Deep learning framework |
| **keras** | 2.14.0 | Neural network API |

### Visualization

| Library | Version | Purpose |
|---------|---------|---------|
| **matplotlib** | 3.8.2 | Static plotting |
| **seaborn** | 0.13.0 | Statistical visualization |
| **plotly** | 5.18.0 | Interactive visualizations |
| **scipy** | 1.11.4 | Scientific computing |

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Sufficient RAM for model training (minimum 8GB recommended)

### Installation

1. **Clone or navigate to the project directory**:
```bash
cd "Cyber Attack Prediction App"
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running the Application

Start the Streamlit app:
```bash
streamlit run app_with_smote.py
```

The application will open in your default browser at `http://localhost:8501`

## 📊 Application Structure

### Pages

#### 🏠 **Home**
- System overview and key features
- Dataset information and statistics
- Getting started guide
- Architecture highlights

#### 📊 **Data Analysis**
- Load the cybersecurity dataset
- Visualize class distribution
- Explore feature correlations
- Statistical feature analysis
- Identify class imbalance issues

#### 🤖 **Model Training**
1. **Data Preprocessing**
   - Feature normalization using StandardScaler
   - Train-test split (80-20)
   - Categorical variable encoding

2. **Random Forest Baseline**
   - 200 decision trees
   - Balanced class weights
   - Hyperparameters optimized for cybersecurity

3. **Deep Learning Model**
   - 4-layer neural network
   - Batch normalization for stability
   - Dropout for regularization
   - Early stopping to prevent overfitting

4. **Data Augmentation with SMOTE + Tomek**
   - SMOTE for synthetic minority oversampling
   - Tomek Links for noise removal
   - Fallback to standard SMOTE if hybrid fails

5. **Enhanced Random Forest**
   - 500 decision trees trained on augmented data
   - Out-of-bag (OOB) score calculation
   - Optimized for balanced dataset performance

#### 📈 **Results**
Three tabs for comprehensive model evaluation:

**Metrics Tab**:
- Model comparison charts
- Accuracy and ROC AUC scores
- Detailed classification reports
- Performance visualization

**Confusion Matrices Tab**:
- Visual confusion matrices for all models
- True positives, false positives, true negatives, false negatives
- Class-specific performance analysis

**Learning Curves Tab**:
- Training history visualizations
- AUC over epochs
- Loss progression
- Overfitting detection

#### 🔮 **Predictions**
- Generate random test samples
- Real-time predictions from all three models
- Ensemble voting system
- Confidence scores and probability distributions
- Model agreement analysis
- Detailed comparison table

## 🧠 Model Architecture

### Random Forest (Baseline)
```
- n_estimators: 200 trees
- max_depth: 20 (prevents overfitting)
- min_samples_split: 5
- min_samples_leaf: 2
- class_weight: 'balanced'
```

### Deep Learning Model
```
Input Layer (Variable dimensions)
    ↓
Dense Layer (256 units, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (30%)
    ↓
Dense Layer (128 units, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (30%)
    ↓
Dense Layer (64 units, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (20%)
    ↓
Dense Layer (1 unit, Sigmoid) → Output
```

### Enhanced Random Forest (SMOTE + Tomek Trained)
```
- n_estimators: 500 trees
- max_depth: 25 (deeper trees on balanced data)
- min_samples_split: 5
- min_samples_leaf: 2
- class_weight: 'balanced_subsample'
- bootstrap: True (bagging for generalization)
- oob_score: True (out-of-bag validation)
```

## 🔄 Data Augmentation Strategy

### SMOTE + Tomek Links

This hybrid approach combines two complementary techniques:

1. **SMOTE (Synthetic Minority Over-sampling Technique)**
   - Generates synthetic samples for the minority class
   - Uses k-nearest neighbors (k=5) to create realistic synthetic data
   - Increases minority class representation

2. **Tomek Links**
   - Identifies ambiguous samples near class boundaries
   - Removes majority class samples that form Tomek links
   - Cleans noisy decision boundaries
   - Improves class separability

**Result**: Balanced, clean dataset with improved class boundary definitions

## 📊 Performance Metrics

The system evaluates models using:

- **Accuracy**: Overall correctness of predictions
- **ROC AUC**: Area under the Receiver Operating Characteristic curve
- **Precision**: False positive rate control
- **Recall**: False negative rate control (attack detection rate)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of prediction types

## 🎯 Ensemble Voting System

The final prediction combines three models through majority voting:

1. **Random Forest** (Baseline)
2. **Deep Learning** (Neural Network)
3. **Augmented Random Forest** (SMOTE + Tomek trained)

**Decision Rule**:
- **Attack Detected**: 2 or 3 models vote for attack
- **Normal Traffic**: 2 or 3 models vote for normal

**Confidence Levels**:
- 🟢 Perfect Agreement: All 3 models agree
- 🟡 Majority Agreement: 2 out of 3 models agree
- 🔴 Model Conflict: Split decision (edge case)

## 🔧 Configuration

Key parameters in `CONFIG` dictionary:

```python
CONFIG = {
    "rf_n_estimators": 200,      # RF trees
    "rf_random_state": 42,        # Reproducibility seed
    "dl_epochs": 50,              # DL training epochs
    "dl_batch_size": 32,          # DL batch size
    "gan_latent_dim": 100,        # GAN latent dimension
    "gan_epochs": 2000,           # GAN training epochs
    "gan_batch_size": 32,         # GAN batch size
    "synthetic_samples": 3000,    # Synthetic samples to generate
}
```

## 💡 Key Implementation Details

### Class Imbalance Handling

1. **Balanced Class Weights**: Penalizes misclassification of minority class
2. **SMOTE**: Generates synthetic minority samples
3. **Tomek Links**: Removes noisy majority class samples
4. **Stratified Split**: Maintains class ratios in train-test split

### Regularization Techniques

- **Dropout**: Prevents overfitting by randomly disabling neurons
- **Batch Normalization**: Stabilizes training and reduces internal covariate shift
- **Early Stopping**: Halts training when validation performance plateaus
- **Learning Rate Reduction**: Decreases learning rate when progress slows

### Data Preprocessing

- **StandardScaler**: Normalizes features to mean=0, std=1
- **Categorical Encoding**: Maps categorical variables to numeric values
- **Missing Value Handling**: Fills NaN values with 0

## 📈 Workflow

```
Load Dataset
    ↓
Data Analysis & Exploration
    ↓
Preprocessing & Normalization
    ↓
Train-Test Split (80-20)
    ↓
┌─────────────────┬──────────────────┬──────────────────┐
│                 │                  │                  │
↓                 ↓                  ↓                  ↓
RF (Baseline)    DL Model      SMOTE + Tomek    Ensemble
                           ↓
                   Augmented RF
    ↓                 ↓                  ↓
Predictions ← ────────┴──────────────────┘
    ↓
Evaluate Metrics
    ↓
Ensemble Voting
    ↓
Final Prediction
```

## 🔐 Security Considerations

- Models trained on labeled intrusion dataset
- Class balancing prevents bias toward majority class
- Ensemble approach reduces individual model weaknesses
- Regular evaluation on test set ensures generalization
- OOB scoring for unbiased performance estimation

## 📁 Project Files

```
Cyber Attack Prediction App/
├── app_with_smote.py          # Main Streamlit application
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Performance Expectations

Based on the architecture and training approach:

- **Baseline RF**: 85-92% accuracy
- **Deep Learning**: 87-94% accuracy
- **Augmented RF**: 89-95% accuracy (typically best)
- **Ensemble**: 90-96% accuracy with high confidence

*Actual results depend on dataset characteristics and class imbalance ratio*

## 🔄 Model Retraining

To retrain models with new data:

1. Replace the dataset URL in the "Data Analysis" page
2. Navigate to "Model Training" page
3. Click "Start Training Pipeline"
4. Models will retrain on new data automatically

## 🐛 Troubleshooting

### Memory Issues
- Reduce `dl_batch_size` if running out of memory
- Train on CPU instead of GPU if needed

### Slow Training
- Reduce number of trees: `rf_n_estimators`
- Decrease `dl_epochs`
- Use GPU acceleration if available

### Dataset Loading Errors
- Ensure dataset URL is accessible
- Check internet connection
- Verify CSV format matches expected structure

## 📚 Further Reading

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow/Keras Guide](https://www.tensorflow.org/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📝 Citation

If you use this system in research or production, please consider citing:

```
Cyber Attack Prediction System
Machine Learning & Deep Learning Ensemble for Network Security
Version 1.0
```

## ⚖️ License

This project is provided as-is for educational and commercial purposes.

## 👨‍💻 Author

Created as an advanced machine learning solution for cybersecurity threat detection.

## 📞 Support

For issues, questions, or improvements, please review the code comments and documentation provided throughout the application.

---

**Last Updated**: December 2024  
**Python Version**: 3.8+  
**Status**: Production Ready
