# 🎮 Cookie Cats — A/B Test Analysis

A professional end-to-end A/B test analysis on real mobile game data,
covering statistical theory, data cleaning, hypothesis testing,
bootstrap simulation, Bayesian inference, and a live interactive dashboard.

---

## 📌 Business Context

Cookie Cats is a massively popular mobile puzzle game developed by Tactile
Entertainment. As players progress through the game, they occasionally
encounter gates that force them to wait or make an in-app purchase to
continue. These gates serve a dual purpose: they generate revenue and
they create natural rest points in the game.

The game team wanted to answer one question:

> **Does moving the first gate from level 30 to level 40 improve
> player retention and engagement?**

This is a classic product experiment — the kind run daily by companies
like King, Supercell, Zynga, and Riot Games.

---

## 🧪 The Experiment

| | Control | Treatment |
|---|---|---|
| **Group** | gate_30 | gate_40 |
| **Change** | Gate at level 30 (original) | Gate moved to level 40 |
| **Users** | 44,700 | 45,489 |
| **Duration** | 14 days |
| **Primary metric** | 7-day retention |
| **Secondary metric** | 1-day retention, rounds played |

---

## 📊 Dataset

**Source:** Real A/B test data from Cookie Cats (via Kaggle)
**Size:** 90,189 players

| Column | Type | Description |
|--------|------|-------------|
| `userid` | int | Unique player identifier |
| `version` | str | gate_30 (control) or gate_40 (treatment) |
| `sum_gamerounds` | int | Rounds played in first 14 days |
| `retention_1` | bool | Still playing after 1 day? |
| `retention_7` | bool | Still playing after 7 days? |

---

## 🧹 Data Cleaning

Before running any statistical test, we cleaned the dataset:

**1. Removed never-played users (3,994 removed)**
Players who installed but never played a single round cannot be
affected by the gate placement. Including them would dilute the
true effect and bias retention rates downward equally for both groups.

**2. Removed extreme outliers (898 removed)**
One player logged 49,854 rounds in 14 days — statistically impossible
for a human player. We removed all users above the 99th percentile
(499 rounds) to eliminate bot activity and data corruption.

**Final clean dataset: 85,335 players**

---

## 📐 Statistical Theory

### 1. Z-Test for Proportions
Used for binary outcomes (retained: yes or no).
Tests whether the difference in retention rates between groups
is larger than what random chance alone could produce.
```
H₀ (null hypothesis)     : retention_30 = retention_40
H₁ (alternative)         : retention_30 ≠ retention_40
Significance level (α)   : 0.05
```

A **p-value below 0.05** means we reject H₀ — the difference is
real, not random noise.

### 2. Confidence Interval
A 95% confidence interval tells us the range of plausible values
for the true difference. If the interval **does not contain 0**,
the effect is statistically significant.

### 3. Bootstrap Simulation (1,000 iterations)
Instead of relying on mathematical assumptions, we resample the
data 1,000 times with replacement and measure the difference each
time. This gives us an empirical distribution of the effect.

**Advantage:** Makes no assumptions about data distribution.
Works even with skewed or non-normal data.

### 4. Bayesian Analysis
Instead of just a p-value, Bayesian inference gives us a direct
probability: *"What is the chance that gate_30 is genuinely better?"*

We model retention as a Beta distribution (the natural distribution
for proportions between 0 and 1) and sample from the posterior
to estimate this probability directly.

**Advantage:** More intuitive than p-values. Tells you what you
actually want to know.

### 5. Minimum Detectable Effect (MDE)
The smallest effect our experiment could reliably detect given
our sample size, significance level, and statistical power.
```
MDE = (z_α + z_β) × √(2 × p × (1-p) / n)
```

If the true effect is smaller than the MDE, our test may miss it
(Type II error / false negative).

---

## 📈 Results

### Frequentist Tests

| Metric | gate_30 | gate_40 | Δ | p-value | Significant? |
|--------|---------|---------|---|---------|--------------|
| 1-Day Retention | 46.24% | 45.70% | +0.54% | 0.111 | ❌ No |
| 7-Day Retention | 19.09% | 18.26% | +0.83% | 0.002 | ✅ Yes |
| Avg Rounds | 46.9 | 46.7 | +0.2 | 0.738 | ❌ No |

### Bootstrap Simulation

| Metric | Mean Diff | 95% CI | P(gate_30 wins) |
|--------|-----------|--------|-----------------|
| 1-Day Retention | +0.55% | [-0.09%, +1.20%] | 94.4% |
| 7-Day Retention | +0.82% | [+0.30%, +1.33%] | 99.8% |

### Bayesian Analysis

| Metric | P(gate_30 better) | P(gate_40 better) |
|--------|-------------------|-------------------|
| 1-Day Retention | 94.33% | 5.67% |
| 7-Day Retention | 99.91% | 0.09% |

---

## 🧠 Business Insights

### Why does gate_30 retain players better?

The result seems counterintuitive — shouldn't players prefer fewer
interruptions? The psychology explains it:

**The gate acts as a natural rest point.**
At level 30, players have invested just enough time to feel
accomplished but not frustrated. The gate gives them a moment
to pause, share their progress on social media, or simply
reflect on how far they've come. This natural pause actually
*increases* the desire to return.

**Moving the gate to level 40 backfires.**
By level 40, players have invested significantly more time.
When they hit a wall at that point, the frustration is much
higher. The gap between effort invested and reward received
feels unfair — and they quit.

**The 7-day signal is stronger than the 1-day signal.**
This tells us the gate placement doesn't affect whether players
come back tomorrow — but it strongly affects whether they stick
around for a week. This is the metric that drives long-term
monetization and lifetime value (LTV).

### What this means at scale

A difference of +0.83% in 7-day retention sounds small.
At Cookie Cats' scale of millions of installs per month:
```
1,000,000 installs × 0.83% = 8,300 additional retained players
per month — all from one gate placement decision.
```

These retained players generate ongoing ad revenue, in-app
purchases, and word-of-mouth growth.

---

## ✅ Recommendation

> **Keep the gate at level 30. Do NOT ship the gate_40 change.**

All three statistical methods agree:
- Frequentist: p = 0.002 (significant at α = 0.05)
- Bootstrap: gate_30 wins in 99.8% of simulations
- Bayesian: 99.91% probability gate_30 is genuinely better

---

## 🏗️ Project Structure
```
ab_testing/
├── cookie_cats.csv       ← raw dataset
├── explore.py            ← initial data exploration
├── analysis.py           ← statistical engine
├── dashboard.py          ← streamlit interactive dashboard
├── eda.ipynb             ← full EDA notebook with visualizations
├── requirements.txt      ← dependencies
├── config.example.py     ← configuration template
└── README.md
```

---

## 🛠️ Tech Stack

- **Python 3** — core language
- **pandas & numpy** — data manipulation
- **scipy & statsmodels** — statistical tests
- **matplotlib & seaborn** — EDA visualizations
- **plotly** — interactive charts
- **streamlit** — live dashboard

---

## 📦 Setup
```bash
git clone https://github.com/hibatallahchmicha/ab-testing-cookie-cats
cd ab-testing-cookie-cats
pip install -r requirements.txt
```

Run EDA notebook:
```bash
jupyter notebook eda.ipynb
```

Run statistical analysis:
```bash
python analysis.py
```

Launch dashboard:
```bash
python -m streamlit run dashboard.py
```

---

## 💡 Key Concepts Practiced

- Hypothesis testing (Z-test, T-test)
- Confidence intervals
- Bootstrap resampling
- Bayesian inference with Beta distributions
- Minimum Detectable Effect calculation
- Data cleaning for A/B tests
- Communicating statistical results to business stakeholders