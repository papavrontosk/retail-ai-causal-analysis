# Methodology

## Why this methodology?

The objective of this project is not simply to estimate whether AI personalization is associated with higher sales, but to approximate a causal relationship using observational panel data.

Retail stores differ substantially in characteristics such as size, customer base, location, and management quality. Consequently, a simple cross-sectional regression would likely produce biased estimates.

To address these challenges, this project combines multiple panel-data estimators, each designed to reduce a different source of bias. Rather than relying on a single specification, the analysis compares several econometric approaches and evaluates the robustness of the estimated AI effect across alternative samples and model assumptions.

---

## Fixed Effects (FE)

The primary specification is a two-way Fixed Effects model including:

- Store fixed effects
- Month fixed effects

This estimator measures how changes in AI adoption **within the same store** are associated with changes in sales while controlling for:

### Store Fixed Effects

These remove permanent store characteristics such as:

- Store size
- Geographic location
- Local customer demographics
- Long-run management quality

### Month Fixed Effects

These capture common shocks affecting every store simultaneously, including:

- Seasonality
- Inflation
- Holidays
- Company-wide promotions
- Macroeconomic conditions

Because identification relies on within-store variation over time, the Fixed Effects estimator provides a substantially stronger basis for causal interpretation than simple comparisons across different stores.

---

## First Differences (FD)

Although Fixed Effects eliminate time-invariant heterogeneity, they may still be influenced by slowly evolving trends.

For this reason, the project also estimates First Difference models.

Instead of comparing sales levels, the FD estimator compares **month-to-month changes**.

Differencing removes all store characteristics that remain constant over time and focuses entirely on short-run movements.

The specification includes:

- Current changes in AI adoption
- One-month lag
- Two-month lag
- One-month lead

These dynamic terms help evaluate:

- Whether the impact appears immediately
- Whether effects accumulate over time
- Whether sales begin changing before AI adoption (possible reverse causality)

To avoid creating artificial observations, first differences are calculated **only between consecutive calendar months**, ensuring that differences are never computed across reporting gaps.

---

## Store-Specific Trends

One FD specification additionally incorporates store-specific trends.

Retail stores rarely evolve in exactly the same way. Some improve steadily because of renovations, stronger local demand, or better management, while others experience gradual declines.

Allowing each store to follow its own linear trend reduces the likelihood that these long-term movements are incorrectly attributed to AI adoption.

Comparing this specification with the standard FD model provides an additional robustness check.

---

## Reduced-T Balanced Panel

The original dataset is naturally **unbalanced**, since stores begin and end reporting at different points in time.

To evaluate whether missing observations influence the results, a balanced panel is constructed by selecting the **12-month calendar window** that maximizes the number of fully observed stores.

This approach improves data consistency and comparability across stores, although it reduces the available time dimension.

---

## Reduced-N Sample

A second robustness exercise keeps the original time span while improving data quality.

Only stores with very few missing reporting months are retained.

Compared with the main sample, this strategy prioritizes reporting quality over sample size, allowing the analysis to assess whether the estimated effects depend on stores with incomplete reporting histories.

---

## Long Difference

The Long Difference estimator compares only the first and last month of the selected balanced window.

Rather than using every monthly observation, it summarizes each store's overall change during the analysis period.

This approach removes permanent store characteristics in a simple and intuitive way, making it a useful first-pass estimator.

However, because intermediate dynamics are ignored, Long Difference generally provides weaker causal evidence than estimators that exploit the full panel structure.

---

## Robustness Strategy

Instead of relying on a single econometric specification, the project evaluates whether conclusions remain consistent across several complementary estimators.

The final comparison includes:

- Fixed Effects (Main Sample)
- First Differences (FD-A)
- First Differences with Store-Specific Trends (FD-B)
- Fixed Effects (Reduced-T)
- FD-B (Reduced-T)
- Fixed Effects (Reduced-N)
- FD-B (Reduced-N)
- Long Difference (Reduced-T)

Consistency across these alternative specifications increases confidence that the estimated relationship is not driven by a particular modeling choice or sample construction.

---

## Business Perspective

From a business perspective, the objective is not to maximize predictive accuracy, but to estimate whether increased adoption of AI personalization genuinely contributes to higher retail sales after accounting for store heterogeneity, reporting quality, and common time effects.

By combining multiple panel-data estimators and robustness checks, the analysis provides stronger evidence for decision-making than a single regression model, helping distinguish genuine operational impact from spurious correlations.
