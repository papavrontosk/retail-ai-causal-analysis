Methodology
Why this methodology?

The objective of this project is not simply to estimate whether AI personalization is associated with higher sales, but to approximate a causal effect using observational panel data.

Because stores differ substantially in size, customer base, location, and management quality, a simple cross-sectional regression would produce biased estimates. The analysis therefore combines several panel-data estimators, each addressing a different source of bias.

Rather than relying on a single model, the project compares multiple specifications and evaluates the robustness of the estimated effect across different assumptions.

Fixed Effects (FE)

The main specification uses a two-way Fixed Effects model with:

Store fixed effects
Month fixed effects

This approach estimates how changes in AI adoption within the same store relate to changes in sales while controlling for permanent differences across stores and common monthly shocks affecting the entire retail network.

Examples of factors removed by store fixed effects include:

Store size
Location
Local customer demographics
Long-run management quality

Month fixed effects account for common events such as:

Seasonality
Inflation
Holidays
Network-wide promotions

Because the identification comes from within-store variation over time, the estimated effect is substantially more informative than a simple comparison between different stores.

First Differences (FD)

Although Fixed Effects remove time-invariant heterogeneity, they may still be sensitive to slowly evolving trends.

For this reason, the project also estimates First Difference models.

Instead of comparing sales levels, the FD estimator compares month-to-month changes.

This transformation naturally removes all store-specific characteristics that remain constant over time and focuses entirely on short-run movements.

The FD specification also includes:

contemporaneous AI adoption changes
lagged changes
lead changes

This dynamic structure allows the analysis to examine:

whether the impact appears immediately,
whether effects accumulate over time,
whether changes in sales precede AI adoption (possible reverse causality).

To avoid introducing artificial observations, first differences are calculated only between consecutive calendar months.

Store-Specific Trends

One First Difference specification additionally includes store-specific trends.

Retail stores rarely evolve identically.

Some stores continuously improve because of renovations, stronger local demand or better management, while others gradually decline.

Allowing each store to follow its own linear trend reduces the risk that these long-run movements are incorrectly attributed to AI adoption.

The comparison between the standard FD model and the trend-adjusted specification provides an additional robustness check.

Reduced-T Balanced Panel

The original dataset is unbalanced because stores enter and exit the reporting system at different times.

To evaluate whether missing observations influence the results, the project constructs a balanced panel by selecting the 12-month calendar window that contains the largest number of fully observed stores.

This design improves data consistency and simplifies comparisons across stores, although it sacrifices part of the available time dimension.

Reduced-N Sample

A second robustness exercise keeps the original time dimension but excludes stores with poor reporting quality.

Only stores with very few missing reporting months are retained.

Compared with the main sample, this approach prioritizes data quality over sample size.

The resulting estimates indicate whether the conclusions depend on stores with incomplete reporting histories.

Long Difference

The Long Difference estimator compares only the first and the last month of the selected balanced window.

Rather than using every monthly observation, it summarizes each store's overall change during the period.

This approach removes permanent store characteristics in a simple and transparent way, making it useful as an initial benchmark.

However, because intermediate dynamics are ignored, Long Difference generally provides less informative causal evidence than panel estimators that exploit the full time series.

Robustness Strategy

Instead of relying on a single econometric specification, the project evaluates consistency across several complementary estimators.

The final comparison includes:

Main Fixed Effects model
First Differences
First Differences with store-specific trends
Fixed Effects on the balanced sample
First Differences on the balanced sample
Fixed Effects on the high-quality reporting sample
First Differences on the high-quality reporting sample
Long Difference estimator

Consistent estimates across these alternative specifications increase confidence that the estimated relationship is not driven by a particular modeling choice or sample construction.

Business Perspective

From a business standpoint, the objective is not to maximize predictive accuracy but to estimate whether increased use of AI personalization genuinely contributes to higher sales after accounting for store heterogeneity, reporting quality and time effects.

The combination of multiple panel-data estimators provides a substantially stronger empirical basis for decision-making than a single regression model, allowing management to assess whether observed sales improvements are likely to reflect a true operational impact of AI adoption rather than differences across stores or reporting artifacts.
