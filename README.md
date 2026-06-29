# Hydraulic System Condition Monitoring ML Classification Pipeline

Predicting the health state of four hydraulic components (cooler, valve, pump, accumulator) from multi-sensor time-series data, using feature engineering and classical ML classifiers.

This project was built as a capstone exploration for applying machine learning to industrial condition monitoring, with an eye toward eventually integrating the resulting models into a conversational, RAG-based diagnostic agent.

## Dataset

**[Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)** — UCI Machine Learning Repository, contributed by ZeMA gGmbH.

> ⚠️ The dataset is **not included in this repo**. Download it directly from the UCI link above (`data.zip`) and extract it into a `raw_data/` folder at the project root before running the notebook.

### Dataset summary

- **2205 cycles** (instances), each representing one 60-second load cycle on a hydraulic test rig
- **17 sensors**, sampled at three different rates:

| Sensor(s) | Quantity | Rate | Values per cycle |
|---|---|---|---|
| PS1–PS6 | Pressure (bar) | 100 Hz | 6000 each |
| EPS1 | Motor power (W) | 100 Hz | 6000 |
| FS1, FS2 | Volume flow (l/min) | 10 Hz | 600 each |
| TS1–TS4 | Temperature (°C) | 1 Hz | 60 each |
| VS1 | Vibration (mm/s) | 1 Hz | 60 |
| CE | Cooling efficiency, virtual (%) | 1 Hz | 60 |
| CP | Cooling power, virtual (kW) | 1 Hz | 60 |
| SE | Efficiency factor (%) | 1 Hz | 60 |

Total: **43,680 raw sensor readings per cycle**.

- **Targets** (`profile.txt`), 5 columns per cycle:
  1. Cooler condition: `3` (close to failure) / `20` (reduced efficiency) / `100` (full efficiency)
  2. Valve condition: `73` (close to failure) / `80` (severe lag) / `90` (small lag) / `100` (optimal)
  3. Internal pump leakage: `0` (none) / `1` (weak) / `2` (severe)
  4. Hydraulic accumulator: `90` (close to failure) / `100` (severely reduced) / `115` (slightly reduced) / `130` (optimal)
  5. Stable flag: `0` (stable) / `1` (static conditions not yet reached) — *not modeled in this version; see Roadmap*

## Approach

1. **Load** : all 17 sensor files + `profile.txt`, verify shapes and alignment against the dataset documentation.
2. **EDA** : visualize class distributions for all 4 targets, and compare raw sensor waveforms across different fault severities to confirm a visible relationship exists before modeling.
3. **Feature engineering** : each sensor's raw waveform per cycle is collapsed into 6 summary statistics: `mean`, `median`, `std`, `min`, `max`, `range`. This reduces 43,680 raw readings per cycle down to **17 sensors × 6 stats = 102 engineered features per cycle** (a ~428x reduction), while preserving signal level, spread, and extremes.
4. **Train/test split** : stratified per target, to preserve class balance given the dataset's natural imbalance (e.g. valve: 1125 optimal vs. 360 per fault grade).
5. **Modeling** : Random Forest and XGBoost classifiers trained per target, evaluated with macro-F1 as the primary metric (not just accuracy, given class imbalance), plus confusion matrices and feature importance plots.

## Results

| Target | Best Model | Macro-F1 | Accuracy |
|---|---|---|---|
| Cooler | Random Forest | 1.000 | 1.000 |
| Valve | Random Forest | 0.971 | 0.977 |
| Pump | XGBoost | 1.000 | 1.000 |
| Accumulator | Random Forest | 0.979 | 0.982 |

Feature importance results align with physical expectations — for example, the cooler classifier's top predictors are `CE` (cooling efficiency) and `CP` (cooling power), the two sensors explicitly designed to summarize cooling performance.

**Note on score interpretation:** these scores are very high, which is expected given this is controlled lab data with deliberately and cleanly induced faults — not noisy field data. This is called out here deliberately, since it's a reasonable question to anticipate from anyone reviewing the results.

## Repository Structure

```
.
├── README.md
├── notebooks/
│   └── hydraulic_classification.ipynb   # full pipeline: load -> EDA -> features -> train -> evaluate
├── raw_data/                            # not tracked in git — download from UCI link above
└── requirements.txt
```

## Setup

```bash
git clone <this-repo-url>
cd <repo-name>

# Download and extract the dataset
# from https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems
# into raw_data/

pip install -r requirements.txt
jupyter notebook notebooks/hydraulic_classification.ipynb
```

### Requirements

- Python 3.10+
- pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, jupyter

## Roadmap

- [ ] Stable flag classifier (separate problem - likely needs transient/trend-based features rather than per-cycle summary stats)
- [ ] Hyperparameter tuning on weaker targets (valve, accumulator)
      
**Potentially:**

- [ ] LangChain RAG layer over hydraulic system documentation, grounding classifier outputs in maintenance guidance
- [ ] Knowledge graph representation (Neo4j) of rig components, linking live classification outputs with historical readings and document knowledge
- [ ] Conversational front end (Streamlit) for natural-language Q&A over rig health

## Citation

If referencing this dataset, please cite:

> Nikolai Helwig, Eliseo Pignanelli, Andreas Schütze, "Condition Monitoring of a Complex Hydraulic System Using Multivariate Statistics," in Proc. I2MTC-2015 — 2015 IEEE International Instrumentation and Measurement Technology Conference, paper PPS1-39, Pisa, Italy, May 11-14, 2015, doi: 10.1109/I2MTC.2015.7151267.

## Author

Smriti Kotiyal
