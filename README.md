# Turbofan Engine Remaining Useful Life (RUL) Predictor

An enterprise Prognostics and Health Management (PHM) platform for forecasting the Remaining Useful Life (RUL) of commercial turbofan engines using multi-channel degradation telemetry from NASA's C-MAPSS dataset.

---

## 1. Executive Summary

In high-reliability industrial systems such as aerospace gas turbines, unplanned equipment downtime leads to severe financial and operational overhead, while premature maintenance wastes valuable component service life. Prognostics and Health Management (PHM) bridges this gap by continuously evaluating sensor telemetry to estimate the number of remaining operating flight cycles before an engine reaches failure thresholds.

This project implements an end-to-end machine learning pipeline and interactive analytical dashboard that ingests 22 sensor and operational telemetry channels, models degradation trajectory, and delivers real-time prognostic forecasts and health state classifications.

---

## 2. Dataset and Telemetry Pipeline

The predictive models are trained on the benchmark NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) FD001 dataset, which simulates realistic run-to-failure degradation trajectories under sea-level conditions.

### 2.1 Raw Telemetry Structure
Each recording consists of 26 columns:
* Engine unit identifier
* Operational cycle index
* 3 operational control settings (altitude, Mach number, throttle resolver angle)
* 21 thermal, pressure, and rotational speed sensor measurements

### 2.2 Target Engineering (Remaining Useful Life)
Ground-truth RUL is formulated by determining the total operational lifespan ($C_{max}$) for each engine unit and computing the cycles remaining at every point in time ($t$):

$$\text{RUL}_t = C_{max} - t$$

Degradation in turbofan components does not progress linearly from cycle 1. During early operation, components exhibit negligible mechanical wear, with sensor measurements hovering within baseline noise bands. Accelerating wear manifests predominantly in the final 50 to 125 cycles as thermal efficiency declines and pressure differentials compound.

### 2.3 Feature Cleaning and Dimension Reduction
Zero-variance telemetry channels contribute no diagnostic information and introduce numerical instability. Automated variance filtering identifies flatlined channels:
* `sensor_18` and `sensor_19` exhibit zero variance and are discarded during data preprocessing.
* The remaining 22 channels (3 operational settings and 19 active sensors) are retained as the training feature matrix.

---

## 3. Telemetry Channel Reference

The 22 active features used for modeling and user interface controls are cataloged below:

| Feature Channel | Parameter Description | Engineering Unit | Degradation Behavior |
| :--- | :--- | :--- | :--- |
| `setting_1` | Altitude / Throttle Resolver Angle | Dimensionless | Baseline operating setpoint |
| `setting_2` | Mach Number | Dimensionless | Baseline operating setpoint |
| `setting_3` | Sea-Level Regime Indicator | Dimensionless | Constant in single-condition regimes |
| `sensor_1` | Total temperature at fan inlet (T2) | Rankine (deg R) | Wind-tunnel ambient reference |
| `sensor_2` | Total temperature at LPC outlet (T24) | Rankine (deg R) | Increases with compressor degradation |
| `sensor_3` | Total temperature at HPC outlet (T30) | Rankine (deg R) | Increases with thermal wear |
| `sensor_4` | Total temperature at LPT outlet (T50) | Rankine (deg R) | Increases with turbine blade wear |
| `sensor_5` | Pressure at fan inlet (P2) | psia | Wind-tunnel pressure reference |
| `sensor_6` | Total pressure in bypass-duct (P15) | psia | Bypass pressure reference |
| `sensor_7` | Total pressure at HPC outlet (P30) | psia | Decreases with compressor wear |
| `sensor_8` | Physical fan speed (Nf) | rpm | Increases to maintain target thrust |
| `sensor_9` | Physical core speed (Nc) | rpm | Increases as turbine efficiency drops |
| `sensor_10` | Engine pressure ratio (P50/P2) | Dimensionless | Core expansion ratio |
| `sensor_11` | Static pressure at HPC outlet (Ps30) | psia | Increases significantly with wear |
| `sensor_12` | Ratio of fuel flow to Ps30 (phi) | pps/psia | Decreases with combustion degradation |
| `sensor_13` | Corrected fan speed (NRf) | rpm | Correlated with sensor 8 |
| `sensor_14` | Corrected core speed (NRc) | rpm | Correlated with sensor 9 |
| `sensor_15` | Bypass Ratio (BPR) | Dimensionless | Increases with internal leakage |
| `sensor_16` | Burner fuel-air ratio (farB) | Dimensionless | Combustion baseline |
| `sensor_17` | Bleed Enthalpy (htBleed) | Dimensionless | Increases with seal wear |
| `sensor_20` | High-Pressure Turbine coolant bleed (W31) | lbm/s | Decreases with cooling degradation |
| `sensor_21` | Low-Pressure Turbine coolant bleed (W32) | lbm/s | Decreases with cooling degradation |

---

## 4. Model Training, Evaluation, and Selection

To establish an evidence-based selection for production deployment, three distinct regression architectures were evaluated under consistent train/test splits (80/20 train-test ratio, random state 42).

### 4.1 Candidate Architectures
1. **Linear Regression (Baseline)**:
   Assumes a linear relationship between input telemetry and remaining cycles. While computationally instantaneous and fully interpretable, it cannot model non-linear thermal wear transitions, leading to underfitting and unrealistic linear extrapolations.
2. **Random Forest Regressor (Ensemble Bagging)**:
   Constructed with 100 unpruned decision trees using variance-reduction splitting. Handles non-linear feature interactions and high-dimensional spaces effectively. However, storing 100 deep, unpruned trees produces a 140 MB artifact file, causing memory strain and exceeding cloud and repository storage boundaries.
3. **XGBoost Regressor (Gradient Boosted Trees)**:
   Trains sequential shallow decision trees that iteratively correct the residual errors of prior estimators. Incorporates L1 and L2 regularization (`reg_alpha`, `reg_lambda`) to penalize complex trees and mitigate collinearity among coupled rotational sensors (such as `sensor_8` and `sensor_13`).

### 4.2 Benchmark Results

| Model Architecture | R-Squared Score | Root Mean Squared Error (RMSE) | Training Duration | Serialization Size |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 0.5696 | 44.34 Cycles | 0.05 seconds | ~2 KB |
| **Random Forest (100 Trees)** | 0.6239 | 41.46 Cycles | 25.00 seconds | ~140 MB |
| **XGBoost (100 Trees)** | 0.6225 | 41.53 Cycles | 0.56 seconds | **~430 KB** |

### 4.3 Why XGBoost Was Selected as the Production Model
While Random Forest and XGBoost achieve virtually identical predictive accuracy on raw evaluation splits (41.46 vs 41.53 cycles RMSE), XGBoost was selected as the champion production model based on core deployment criteria:
* **Footprint Reduction**: XGBoost compiles into an artifact of approximately 430 KB compared to 140 MB for Random Forest (a 99.7% reduction in storage and memory load). This permits deployment without Git LFS or out-of-memory errors on serverless cloud platforms.
* **Inference Latency**: Vectorized C++ execution computes individual cycle forecasts in sub-millisecond runtimes, ensuring instantaneous user interface reactivity.
* **Regularization vs Collinearity**: Gas turbine sensors exhibit substantial multicollinearity (e.g. core speed and corrected core speed). XGBoost's built-in tree regularization prevents over-weighting redundant channels.

---

## 5. User Interface Architecture (Streamlit)

The web interface is designed with Streamlit, prioritizing engineer usability, rapid assessment, and responsive layouts.

### 5.1 Balanced Grid Layout
Industrial telemetry requires reviewing numerous sensors at once. To avoid horizontal scrolling tables, all 22 features are arranged in a dynamic 4-column input grid where each measurement can be inspected and updated directly.

### 5.2 Session State and Widget Synchronization
In Streamlit, binding widgets to independent state dictionaries without key alignment causes widgets to overwrite session values with stale state upon rerun. The input pipeline explicitly aligns widget keys (`input_grid_{col}`) with session cache to ensure seamless preset selection and user manual inputs without input reversion.

### 5.3 Calibrated Operating Presets
To facilitate evaluation, the platform includes pre-calibrated profiles representing verified stages along the NASA C-MAPSS degradation curve:
* **Healthy Profile (Nominal State)**: High bypass ratio and efficient pressure differentials yielding approximately 204 remaining cycles.
* **Moderate Wear Profile (Mid-Life Degradation)**: Elevated turbine outlet temperatures and static pressure increases yielding approximately 90 remaining cycles.
* **Critical Profile (Imminent Failure)**: Severe thermal spikes and degraded coolant bleed values yielding approximately 8 remaining cycles.
* **Custom (Manual)**: Enables maintenance engineers to enter telemetry from active flights or test benches.

### 5.4 Prognostic Risk Classification Logic
Forecasts are mapped to standard maintenance triage tiers:
* **Nominal / Healthy**: Estimated RUL > 120 Cycles. Normal operation; continue routine line inspections.
* **Moderate Wear**: 40 < Estimated RUL <= 120 Cycles. Component degradation observed; schedule maintenance within upcoming fleet window.
* **Critical / Service Required**: Estimated RUL <= 40 Cycles. Immediate servicing required; ground aircraft engine for overhaul.

---

## 6. Repository File Layout

```text
RUL-Predictor/
├── app.py                  Main Streamlit web application
├── turbofan_model.pkl      Serialized production XGBoost regressor (~430 KB)
├── feature_cols.pkl        Serialized list of active 22 input feature column names
├── requirements.txt        Python package dependencies
└── README.md               Project documentation and reference manual
```

---

## 7. Installation and Quickstart

### 7.1 Prerequisites
Python version 3.9 through 3.14 on Linux, macOS, or Windows.

### 7.2 Environment Setup
Clone the repository and install required packages:

```bash
git clone https://github.com/vivi-dot-exe/RUL-Predictor.git
cd RUL-Predictor
pip install -r requirements.txt
```

### 7.3 Launch the Application
Run the Streamlit server:

```bash
streamlit run app.py
```

The application will be accessible in your local browser at `http://localhost:8501`.

---

## 8. Current Project Status and Ongoing Development

This project is actively maintained and currently undergoing continuous development and iterative feature expansion. 

Current areas of ongoing work include:
* Integration of multi-condition C-MAPSS subsets (FD002, FD004) to support multi-altitude flight envelope normalization.
* Implementation of local feature attribution using TreeSHAP to provide engineers with transparent, per-sensor risk contributions behind every predicted cycle score.
* Expanded batch evaluation tools for processing historical fleet CSV logs and plotting dynamic fleet degradation curves over multi-year timelines.
