# Uncertainty-Aware Glucose Forecasting and Risk-Aware Decision Support using Deep Learning

## 📌 Overview

This project presents an integrated machine learning framework for **blood glucose forecasting under uncertainty and risk-aware decision-making**.

Instead of generating only a single predicted glucose value, the system produces a **prediction interval** that represents a range in which the future glucose value is expected to lie. These uncertainty-aware predictions are then passed to a **utility-based decision-making framework** that evaluates possible actions and recommends the action with the highest utility.

The overall framework combines:

* Deep learning-based time-series forecasting
* Prediction interval estimation
* Custom confidence-based / Tube Loss
* Uncertainty quantification using **PICP** and **MPIW**
* Artificial Neural Network (ANN)-based utility modeling
* Sigmoid-based smooth utility functions
* Risk-aware action selection
* Patient-specific glucose analysis

The project was evaluated using real-world data from **25 Type 1 Diabetes patients** from the HUPA-UCM Diabetes Dataset.

---

## 🎯 Problem Statement

Blood glucose levels are highly dynamic and influenced by several physiological and behavioral factors such as meals, insulin administration, physical activity, and other patient-specific variations.

Traditional glucose prediction approaches often generate **single-point forecasts**. Although these predictions provide an estimate of future glucose levels, they do not explicitly communicate how uncertain the prediction is.

This becomes particularly important in healthcare applications because a prediction that is slightly wrong in a low-risk situation may be acceptable, while the same uncertainty near hypoglycemic or hyperglycemic ranges can have significantly greater consequences.

Another limitation is that traditional decision-making systems often rely on **fixed threshold-based rules**. Such rules can introduce abrupt transitions between actions even though physiological risk changes gradually.

Therefore, the project addresses the following problem:

> **How can future blood glucose levels be predicted while explicitly representing uncertainty, and how can these uncertain predictions be translated into reliable and risk-aware actions?**

The proposed solution combines **prediction interval-based forecasting with utility-driven decision-making**.

---

## 🎯 Objectives

The major objectives of the project were:

1. Develop a deep learning framework for blood glucose time-series forecasting.
2. Generate **prediction intervals instead of only point predictions**.
3. Quantify prediction uncertainty using **Prediction Interval Coverage Probability (PICP)** and **Mean Prediction Interval Width (MPIW)**.
4. Design a custom loss function capable of balancing interval coverage and interval width.
5. Develop an ANN-based utility model for evaluating possible actions.
6. Use sigmoid functions to create smooth and continuous utility boundaries.
7. Select actions using a **risk-aware utility maximization approach**.
8. Evaluate the framework on real-world data from multiple Type 1 Diabetes patients.

---

# 📊 Dataset

The project uses the **HUPA-UCM Diabetes Dataset**, containing real-world medical data collected from **25 patients with Type 1 Diabetes Mellitus (T1DM)**.

The data was collected through a clinical study involving:

* Hospital Universitario Príncipe de Asturias, Spain
* Universidad Complutense de Madrid

Each patient was monitored for a minimum of **14 days** under free-living conditions.

### Data Sources

The dataset combines information from multiple sensing and monitoring systems, including:

* Continuous Glucose Monitoring (CGM)
* Insulin delivery systems
* Fitbit wearable devices

### Main Features

| Feature             | Description                      |
| ------------------- | -------------------------------- |
| Time                | Timestamp of observation         |
| Glucose             | Blood glucose level in mg/dL     |
| Heart Rate          | Heart rate from wearable devices |
| Calories Burned     | Energy expenditure               |
| Steps               | Physical activity                |
| Basal Insulin       | Continuous insulin infusion      |
| Bolus Insulin       | Administered insulin doses       |
| Carbohydrate Intake | Meal carbohydrate intake         |

After preprocessing, observations were primarily aligned to **5-minute intervals**.

---

# ⚙️ Methodology

The complete system can be viewed as two connected components:

```text
                    Real-World Patient Data
                             │
                             ▼
                    Data Preprocessing
                             │
                             ▼
                    Time-Series Features
                             │
                             ▼
              ┌───────────────────────────┐
              │ Deep Learning Models      │
              │ LSTM / GRU / TCN          │
              └───────────────────────────┘
                             │
                             ▼
                Prediction Intervals
                  Lower Bound / Upper Bound
                             │
                             ▼
                  Uncertainty Analysis
                     PICP / MPIW
                             │
                             ▼
              ┌───────────────────────────┐
              │ ANN Utility Model         │
              │ + Sigmoid Functions       │
              └───────────────────────────┘
                             │
                             ▼
                    Utility Scores
                             │
                             ▼
                  Risk-Aware Action
                       Selection
```

---

# 🧹 Data Preprocessing

The preprocessing pipeline was designed to clean the glucose time series while preserving clinically relevant fluctuations.

### 1. Temporal Sorting

Data was first sorted chronologically to preserve the temporal structure of the glucose measurements.

### 2. Missing Value Handling

Missing observations were handled using interpolation to maintain continuity in the time series.

### 3. Outlier Handling

Outliers were addressed through percentile-based clipping and medically valid glucose range restrictions.

The glucose values were restricted to:

**50–300 mg/dL**

### 4. Noise Reduction

Several techniques were explored, including:

* Median filtering
* Moving-average filtering
* Threshold-based spike detection

However, aggressive filtering was avoided because excessive smoothing could remove genuine physiological variations.

The final preprocessing pipeline therefore used **light smoothing** to preserve natural glucose fluctuations.

### 5. Normalization

**Z-score normalization** was applied to transform the input data to zero mean and unit variance.

This helped improve:

* Training stability
* Model convergence
* Deep learning performance

### 6. Lag Features

Historical glucose observations were incorporated using lag features.

The selected lag values were:

```text
1   → previous 5 minutes
2   → previous 10 minutes
3   → previous 15 minutes
6   → previous 30 minutes
9   → previous 45 minutes
12  → previous 60 minutes
```

These lag features allow the model to capture temporal dependencies in glucose dynamics.

---

# 🧠 Deep Learning Models

Three deep learning architectures were evaluated for prediction interval forecasting:

### LSTM

Long Short-Term Memory networks were used to model temporal dependencies in glucose time-series data.

### GRU

Gated Recurrent Units were evaluated as an alternative recurrent architecture capable of learning temporal patterns with a comparatively simpler structure.

### TCN

Temporal Convolutional Networks were also evaluated for capturing temporal patterns in the glucose signal.

The models were trained to generate **prediction intervals rather than single-point predictions**.

---

# 📦 Prediction Interval Forecasting

A key component of this project is the transition from point prediction to **interval prediction**.

Instead of predicting:

```text
Glucose = 145 mg/dL
```

the model produces:

```text
Lower Bound ───────── Upper Bound
      │                    │
      └──── Expected ──────┘
          Glucose Range
```

This provides additional information about the uncertainty associated with the prediction.

A reliable prediction interval should:

* Contain most of the actual glucose observations.
* Remain sufficiently narrow to be useful.

This creates a trade-off between **coverage** and **interval width**.

---

# 📏 Uncertainty Quantification

Two primary metrics were used to evaluate the prediction intervals.

## Prediction Interval Coverage Probability (PICP)

PICP measures the proportion of actual glucose values that fall inside the predicted interval.

Higher PICP indicates better coverage and reliability.

---

## Mean Prediction Interval Width (MPIW)

MPIW measures the average width of the generated prediction intervals.

Lower MPIW is generally preferred because excessively wide intervals provide less useful information.

Therefore, the objective is:

> **High PICP + Low MPIW**

This allows the model to provide reliable uncertainty estimates without generating unnecessarily broad intervals.

---

# 🔥 Custom Tube Loss

A custom confidence-based loss function was developed to directly optimize prediction intervals.

The loss consists of three components:

### 1. Outside Loss

Penalizes predictions when the actual glucose value falls outside the predicted interval.

This encourages the model to increase coverage.

### 2. Inside Loss

Controls how observations are distributed within the prediction interval.

### 3. Width Penalty

Penalizes excessively wide prediction intervals.

The total loss can be represented as:

```text
Ltotal = Loutside + Linside + Lwidth
```

The width penalty is:

```text
Lwidth = δ(U - L)
```

where:

* `U` = upper prediction bound
* `L` = lower prediction bound
* `δ` = width penalty coefficient

This loss therefore attempts to balance two competing objectives:

```text
Reliable Coverage
       ↕
Narrow Prediction Intervals
```

---

# ⚙️ Training Configuration

The models were trained using the following configuration:

| Parameter                | Value |
| ------------------------ | ----: |
| Training Data            |   80% |
| Calibration / Validation |   10% |
| Testing                  |   10% |
| Window Size              |    24 |
| Batch Size               |    64 |
| Epochs                   |   100 |
| Learning Rate            | 0.001 |
| Delta                    |  0.01 |

Early stopping was used to reduce unnecessary computation and help prevent overfitting.

---

# 📈 Prediction Interval Results

The three deep learning architectures were evaluated using PICP and MPIW.

| Model            |       PICP |     MPIW |
| ---------------- | ---------: | -------: |
| LSTM + Tube Loss | **0.9833** |     5.99 |
| GRU + Tube Loss  | **0.9802** | **4.30** |
| TCN + Tube Loss  |     0.9765 | **3.64** |

### Interpretation

The models achieved high prediction interval coverage across the experiments.

The **GRU + Tube Loss** model achieved:

```text
PICP = 0.9802
MPIW = 4.30
```

This provided a strong balance between coverage and interval width.

The TCN achieved the narrowest interval:

```text
MPIW = 3.64
```

while maintaining a PICP of:

```text
0.9765
```

These results demonstrate that the proposed approach can generate reliable prediction intervals while maintaining relatively tight uncertainty bounds.

---

# 🧮 Utility-Based Decision Making

Prediction alone is not the final objective of the project.

The next stage converts predicted glucose conditions into **action recommendations**.

The system defines four possible actions:

```text
1. No Action
2. Take Sugar
3. Monitor
4. Insulin
```

The initial action framework was based on glucose ranges:

| Glucose Range   | Example Action |
| --------------- | -------------- |
| Below 70 mg/dL  | Take Sugar     |
| 70–140 mg/dL    | No Action      |
| 140–180 mg/dL   | Monitor        |
| Above 180 mg/dL | Insulin        |

These ranges were used as an initial guiding framework rather than as a rigid rule-based decision system.

---

# 🧠 ANN-Based Utility Model

An **Artificial Neural Network (ANN)** was designed to learn the relationship between:

```text
Predicted Glucose
        +
Prediction Interval
        ↓
Utility Scores
        ↓
Possible Actions
```

Instead of simply applying fixed thresholds, the ANN learns a mapping from glucose conditions to utility values.

The utility model allows the system to evaluate every possible action and select the action with the highest utility.

---

# 📉 Sigmoid-Based Utility Function

Sigmoid functions were used to create smooth transitions between different utility regions.

The sigmoid function maps values into a bounded range:

```text
0 → 1
```

This makes the resulting utility scores easier to compare and interpret.

The use of sigmoid functions also avoids abrupt transitions associated with conventional step-based rules.

Conceptually:

```text
Traditional:

Utility
  │       ┌────────
  │       │
  │───────┘
  └──────────────── Glucose


Proposed:

Utility
  │          ______
  │        /
  │      /
  │_____/
  └──────────────── Glucose
```

The learned utility function therefore provides a smoother representation of how risk and desirability change with glucose levels.

---

# 🎯 Action Selection

For every prediction instance:

1. The deep learning model generates a glucose prediction interval.
2. The predicted glucose condition is passed to the ANN.
3. Utility scores are generated for all possible actions.
4. The action with the **highest utility score** is selected.
5. The prediction interval is also examined to provide contextual information about glucose variability and risk.

The decision process can be summarized as:

```text
Glucose Time Series
        ↓
Deep Learning Model
        ↓
Prediction Interval
        ↓
ANN Utility Model
        ↓
Utility Scores
        ↓
Highest Utility
        ↓
Recommended Action
```

This creates a decision-making system that considers both **predicted glucose behavior and uncertainty**.

---

# 🛡️ Risk-Aware Decision Making

A major focus of the project is **risk-aware decision-making**.

Rather than treating every prediction equally, the framework considers the potential consequences associated with different glucose states.

For example:

```text
Low Glucose
     ↓
Higher Risk
     ↓
Sugar-related Action

Normal Glucose
     ↓
Lower Risk
     ↓
No Action

Slightly High Glucose
     ↓
Monitoring
     
High Glucose
     ↓
Higher Risk
     ↓
Insulin-related Action
```

The final action is selected using the learned utility scores while also considering the predicted interval.

---

# 👥 Multi-Patient Evaluation

The framework was applied to glucose data from all **25 patients** in the HUPA-UCM dataset.

For each patient:

* Preprocessed glucose data was used as input.
* Prediction intervals were generated.
* The predicted bounds were converted back to the original glucose scale.
* Actual glucose values were compared against the predicted intervals.
* Utility scores were generated for possible actions.
* The highest-utility action was selected.

The resulting visualizations showed how prediction intervals dynamically changed according to glucose variability.

Intervals became:

* **Wider** during periods of greater glucose variability.
* **Narrower** during relatively stable periods.

This demonstrates the ability of the framework to represent changing uncertainty over time.

---

# 📊 Outcomes

The project achieved several important outcomes.

### Prediction Performance

The models achieved high prediction interval coverage:

```text
LSTM + Tube Loss → PICP 0.9833
GRU + Tube Loss  → PICP 0.9802
TCN + Tube Loss  → PICP 0.9765
```

The GRU model achieved:

```text
PICP = 0.9802
MPIW = 4.30
```

providing a strong balance between coverage and interval width.

### Uncertainty Estimation

Instead of relying exclusively on point predictions, the framework generates prediction intervals that explicitly represent uncertainty.

### Decision Support

The predicted glucose conditions were translated into utility scores for multiple possible actions.

### Smooth Decision Boundaries

The sigmoid-based utility model provided continuous transitions instead of abrupt threshold-based changes.

### Risk-Aware Recommendations

The final decision was based on the action with the highest learned utility while also considering the predicted glucose interval.

### Interpretability

Utility curves and prediction interval plots were used to analyze and interpret model behavior.

---

# 🔬 Key Technical Contributions

The main technical components developed in this project are:

* Real-world glucose time-series preprocessing
* Lag-based temporal feature engineering
* LSTM-based interval forecasting
* GRU-based interval forecasting
* TCN-based interval forecasting
* Custom Tube Loss / confidence-based loss
* Prediction Interval Coverage Probability (PICP)
* Mean Prediction Interval Width (MPIW)
* ANN-based utility modeling
* Sigmoid-based smooth utility functions
* Utility-based action selection
* Risk-aware decision support
* Multi-patient glucose interval analysis

---

# 🧪 Technologies & Concepts

### Machine Learning

* Deep Learning
* Artificial Neural Networks
* Time-Series Forecasting
* Uncertainty Quantification
* Utility-Based Decision Making

### Deep Learning Architectures

* LSTM
* GRU
* TCN

### Mathematical / Statistical Concepts

* Prediction Intervals
* PICP
* MPIW
* Custom Loss Functions
* Sigmoid Functions
* Utility Functions
* Risk-Aware Decision Making

### Data Processing

* Time-series preprocessing
* Interpolation
* Normalization
* Outlier handling
* Lag feature engineering

---

# 📁 Suggested Repository Structure

```text
├── README.md
├── data/
│   └── README.md
│
├── preprocessing/
│   └── preprocessing.py
│
├── models/
│   ├── lstm.py
│   ├── gru.py
│   └── tcn.py
│
├── losses/
│   └── tube_loss.py
│
├── uncertainty/
│   └── prediction_intervals.py
│
├── utility/
│   ├── utility_model.py
│   └── sigmoid_utility.py
│
├── decision/
│   └── action_selection.py
│
├── visualization/
│   ├── prediction_intervals.py
│   └── utility_curves.py
│
└── results/
    └── figures/
```

*The structure above is a recommended organization for the repository; it should be adjusted to match the actual files in your implementation.*

---

# 🚀 Overall Framework

The complete proposed framework can be summarized as:

```text
                   HUPA-UCM Dataset
                          │
                          ▼
                Data Preprocessing
                          │
                          ▼
                 Lag Feature Creation
                          │
                          ▼
              ┌─────────────────────┐
              │ Deep Learning       │
              │ LSTM / GRU / TCN    │
              └─────────────────────┘
                          │
                          ▼
                Prediction Intervals
                          │
                          ▼
                 PICP / MPIW Analysis
                          │
                          ▼
                 Uncertainty-Aware
                    Predictions
                          │
                          ▼
              ┌─────────────────────┐
              │ ANN Utility Model   │
              │ + Sigmoid Functions │
              └─────────────────────┘
                          │
                          ▼
                   Utility Scores
                          │
                          ▼
              Highest Utility Action
                          │
                          ▼
                Risk-Aware Decision
```

---

# 📌 Conclusion

This project demonstrates an integrated approach to **blood glucose forecasting and decision support under uncertainty**.

The use of prediction intervals allows the system to communicate uncertainty instead of relying solely on point estimates. The custom Tube Loss helps balance prediction interval coverage and width, while the PICP and MPIW metrics provide quantitative evaluation of uncertainty quality.

The second stage of the framework uses an ANN-based utility model with sigmoid functions to translate glucose predictions and their associated uncertainty into actionable utility scores.

The final action is selected based on the highest utility, creating a **risk-aware and adaptive decision-making framework** rather than relying entirely on fixed threshold-based rules.

Overall, the project demonstrates how **deep learning, uncertainty quantification, and utility-based decision theory can be combined to build interpretable decision-support systems for real-world healthcare applications.**

---

# 📚 References

1. Shayan Kiyani, George Pappas, Aaron Roth, Hamed Hassani — *Decision Theoretic Foundations for Conformal Prediction: Optimal Uncertainty Quantification for Risk-Averse Agents*

2. J. Ignacio Hidalgoa, Jorge Alvarado, Marta Botella, Aranzazu Aramendi, J. Manuel Velasco, Oscar Garnica — *HUPA-UCM Diabetes Dataset*

3. *Ensemble Tube Loss Networks for Probabilistic Electricity Load Forecasting*

---

# 👨‍💻 Project Information

**Project:** Uncertainty-Aware Glucose Forecasting and Risk-Aware Decision Support using Deep Learning

**Dataset:** HUPA-UCM Diabetes Dataset

**Patients:** 25 Type 1 Diabetes patients

**Models:** LSTM, GRU, TCN

**Uncertainty Method:** Prediction Interval Forecasting using Tube Loss

**Decision Model:** ANN + Sigmoid-based Utility Function

**Evaluation Metrics:** PICP, MPIW

**Primary Goal:** Reliable glucose forecasting combined with uncertainty-aware, risk-aware action selection.
