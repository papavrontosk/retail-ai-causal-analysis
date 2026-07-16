# Retail AI Personalization Impact Analysis

### Causal Impact Evaluation using Panel Data Econometrics

---

## Project Overview

AI-powered personalization has become a core capability for modern retailers, but measuring its business impact is far from straightforward. Stores differ in location, management quality, customer behavior, digital maturity and many other characteristics that can bias simple before-and-after comparisons.

This project investigates whether increasing the usage of an AI personalization engine leads to higher retail sales. Rather than relying on descriptive analytics or predictive models, the analysis applies panel data econometric methods to estimate the causal effect of AI adoption while controlling for both store-specific characteristics and common time shocks.

The project demonstrates an end-to-end analytical workflow, including data engineering, panel construction, exploratory analysis, econometric modeling, robustness checks and business interpretation.

---

# Business Problem

A national retail company gradually rolled out an AI-powered personalization engine across its store network.

Management wants to understand:

> **Does increasing the share of AI-assisted transactions generate measurable sales growth?**

Because adoption occurs at different points in time and stores are observed repeatedly, panel data methods provide an appropriate framework for isolating the effect of AI usage from other factors affecting sales.

---

# Dataset

The analysis combines three independent datasets.

| Dataset | Description |
|----------|-------------|
| **store_info.csv** | Static store characteristics (location, format, floor space, management quality, operating period) |
| **algo_rollout.csv** | AI rollout schedule, treatment status and activation dates |
| **store_month_metrics.csv** | Monthly operational and commercial KPIs at the store level |

After merging the datasets, a longitudinal store-month panel is constructed.

---

# Analytical Workflow

## 1. Data Engineering

- Import raw datasets
- Merge multiple data sources
- Parse monthly dates
- Validate operating periods
- Create analysis-ready panel structure

---

## 2. Panel Construction

A transparent filtering pipeline is implemented to construct the final analysis sample.

Filtering steps include:

- keeping observations within each store's operating window
- removing incomplete monthly reports
- excluding stores with insufficient reporting history
- validating treatment history for treated stores

The project explicitly tracks the number of stores and observations remaining after every filtering step.

---

## 3. Exploratory Data Analysis

Exploratory analysis includes

- summary statistics
- missing data inspection
- panel diagnostics
- monthly trends
- AI adoption patterns
- sales evolution over time

Stores are classified into:

- Never Treated
- Not Yet Live
- Live

to visualize how AI adoption evolves throughout the rollout period.

---

## 4. Econometric Analysis

Several complementary panel estimators are implemented to estimate the causal effect of AI utilization.

### Two-Way Fixed Effects

Controls for

- store-specific heterogeneity
- common monthly shocks

using clustered standard errors at the store level.

---

### First Difference Models

A strict first-difference dataset is constructed to avoid differencing across gaps in the reporting calendar.

Dynamic specifications include

- contemporaneous effects
- lagged treatment effects
- lead effects (pre-trend diagnostics)

---

### Robustness Checks

To evaluate the stability of the estimates, the analysis is repeated using two alternative panel constructions.

#### Reduced-T Balanced Panel

Selects the 12-month window maximizing the number of fully observed stores.

#### Reduced-N Sample

Keeps only stores with high reporting completeness.

---

### Long Difference Estimation

A long-difference specification compares only the first and last observations within the balanced panel window, providing an alternative long-run estimate of the treatment effect.

---

# Methods Used

| Category | Techniques |
|-----------|------------|
| Data Engineering | Data cleaning, merging, feature construction |
| Exploratory Analysis | Time-series visualization, descriptive statistics |
| Econometrics | Two-Way Fixed Effects |
| Econometrics | First Differences |
| Econometrics | Dynamic Distributed Lag Models |
| Robustness | Reduced-T Analysis |
| Robustness | Reduced-N Analysis |
| Robustness | Long Difference Estimation |

---

# Repository Structure

```
.
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── analysis.ipynb
│
├── src/
│   ├── data_preparation.py
│   ├── analysis_sample.py
│   ├── visualization.py
│   ├── fe_models.py
│   ├── fd_models.py
│   ├── reduced_T.py
│   ├── reduced_N.py
│   ├── long_difference.py
│   ├── comparison_table.py
│   └── utils.py
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── report/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Statsmodels
- linearmodels
- SciPy

---

# Key Skills Demonstrated

### Data Analytics

- Data Cleaning
- Data Wrangling
- Feature Engineering
- Exploratory Data Analysis

### Econometrics

- Panel Data Analysis
- Fixed Effects Models
- First Difference Estimation
- Dynamic Treatment Effects
- Cluster-Robust Inference

### Statistical Modeling

- Causal Inference
- Robustness Analysis
- Longitudinal Data Analysis

### Business Analytics

- KPI Analysis
- Retail Analytics
- AI Impact Evaluation
- Decision Support

---

# Results

The project produces

- cleaned panel datasets
- exploratory visualizations
- econometric model outputs
- robustness analyses
- model comparison tables
- business-focused interpretation of AI impact

---

# Why This Project

This project showcases practical experience in applying causal inference techniques to real-world business problems. Rather than focusing solely on predictive performance, it demonstrates how panel data methods can be used to estimate the business impact of operational interventions while accounting for unobserved heterogeneity, reporting gaps and dynamic treatment effects.

The workflow emphasizes reproducibility, transparent data preparation and interpretable results—skills commonly required in Analytics, Data Science, Consulting and Economic Research roles.

---

# Disclaimer

The datasets used in this repository are synthetic and are intended exclusively for demonstrating analytical and econometric methodologies.
