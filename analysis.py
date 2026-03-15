import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# LOAD & CLEAN
# ─────────────────────────────────────────

def load_and_clean():
    df = pd.read_csv("data/cookie_cats.csv")

    print("🧹 Cleaning data...")
    original = len(df)

    # Remove users who never played
    df = df[df["sum_gamerounds"] > 0]
    print(f"   Removed {original - len(df):,} users with 0 rounds")

    # Remove extreme outliers (above 99th percentile)
    q99 = df["sum_gamerounds"].quantile(0.99)
    df  = df[df["sum_gamerounds"] <= q99]
    print(f"   Removed outliers above {q99:.0f} rounds")

    print(f"   Final dataset: {len(df):,} users\n")
    return df


# ─────────────────────────────────────────
# FREQUENTIST TEST — Z-TEST FOR PROPORTIONS
# ─────────────────────────────────────────

def z_test(df, metric):
    """
    Test if retention rates are significantly different.
    Used for binary outcomes (retained: yes/no).
    """
    control   = df[df["version"] == "gate_30"][metric]
    treatment = df[df["version"] == "gate_40"][metric]

    n1, n2   = len(control), len(treatment)
    p1, p2   = control.mean(), treatment.mean()

    # Pooled proportion
    p_pool   = (control.sum() + treatment.sum()) / (n1 + n2)
    se       = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z_stat   = (p1 - p2) / se
    p_value  = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # 95% confidence interval for difference
    diff     = p1 - p2
    margin   = 1.96 * np.sqrt((p1*(1-p1)/n1) + (p2*(1-p2)/n2))
    ci_low   = diff - margin
    ci_high  = diff + margin

    return {
        "metric":     metric,
        "gate_30":    round(p1 * 100, 3),
        "gate_40":    round(p2 * 100, 3),
        "difference": round(diff * 100, 3),
        "z_stat":     round(z_stat, 4),
        "p_value":    round(p_value, 6),
        "ci_low":     round(ci_low * 100, 3),
        "ci_high":    round(ci_high * 100, 3),
        "significant": p_value < 0.05,
        "winner":     "gate_30" if p1 > p2 else "gate_40"
    }


# ─────────────────────────────────────────
# T-TEST — ROUNDS PLAYED
# ─────────────────────────────────────────

def t_test_rounds(df):
    """
    Test if average rounds played differs between groups.
    Used for continuous outcomes.
    """
    control   = df[df["version"] == "gate_30"]["sum_gamerounds"]
    treatment = df[df["version"] == "gate_40"]["sum_gamerounds"]

    t_stat, p_value = stats.ttest_ind(control, treatment)
    diff = control.mean() - treatment.mean()

    return {
        "metric":      "avg_rounds_played",
        "gate_30":     round(control.mean(), 2),
        "gate_40":     round(treatment.mean(), 2),
        "difference":  round(diff, 2),
        "t_stat":      round(t_stat, 4),
        "p_value":     round(p_value, 6),
        "significant": p_value < 0.05,
        "winner":      "gate_30" if diff > 0 else "gate_40"
    }


# ─────────────────────────────────────────
# BOOTSTRAP SIMULATION
# ─────────────────────────────────────────

def bootstrap(df, metric, n_iterations=1000):
    """
    Resample the data 1000 times to estimate how stable
    our results are. More robust than a single test.
    """
    print(f"🔄 Running bootstrap simulation ({n_iterations} iterations)...")

    control   = df[df["version"] == "gate_30"][metric].values
    treatment = df[df["version"] == "gate_40"][metric].values
    diffs     = []

    for _ in range(n_iterations):
        c_sample = np.random.choice(control,   size=len(control),   replace=True)
        t_sample = np.random.choice(treatment, size=len(treatment), replace=True)
        diffs.append(c_sample.mean() - t_sample.mean())

    diffs = np.array(diffs)
    prob_control_wins = (diffs > 0).mean()

    return {
        "metric":             metric,
        "mean_diff":          round(np.mean(diffs) * 100, 4),
        "ci_low":             round(np.percentile(diffs, 2.5) * 100, 4),
        "ci_high":            round(np.percentile(diffs, 97.5) * 100, 4),
        "prob_control_wins":  round(prob_control_wins * 100, 1),
        "diffs":              diffs.tolist()
    }


# ─────────────────────────────────────────
# BAYESIAN ANALYSIS
# ─────────────────────────────────────────

def bayesian_analysis(df, metric):
    """
    Instead of just p-value, estimate the probability
    that gate_30 is truly better than gate_40.
    """
    control   = df[df["version"] == "gate_30"][metric]
    treatment = df[df["version"] == "gate_40"][metric]

    # Beta distribution parameters
    # Prior: Beta(1,1) = uniform — we assume nothing upfront
    alpha_c = control.sum()   + 1
    beta_c  = (~control).sum() + 1
    alpha_t = treatment.sum() + 1
    beta_t  = (~treatment).sum() + 1

    # Sample from posterior distributions
    samples_c = np.random.beta(alpha_c, beta_c, 100000)
    samples_t = np.random.beta(alpha_t, beta_t, 100000)

    prob = (samples_c > samples_t).mean()

    return {
        "metric":                  metric,
        "prob_gate30_better":      round(prob * 100, 2),
        "prob_gate40_better":      round((1 - prob) * 100, 2),
        "gate30_posterior_mean":   round(alpha_c / (alpha_c + beta_c) * 100, 3),
        "gate40_posterior_mean":   round(alpha_t / (alpha_t + beta_t) * 100, 3),
        "samples_control":         samples_c.tolist()[:5000],
        "samples_treatment":       samples_t.tolist()[:5000],
    }


# ─────────────────────────────────────────
# MINIMUM DETECTABLE EFFECT
# ─────────────────────────────────────────

def calculate_mde(df, metric):
    """
    What's the smallest effect we could reliably detect
    given our sample size?
    """
    n        = len(df) // 2
    p        = df[metric].mean()
    alpha    = 0.05
    power    = 0.80
    z_alpha  = stats.norm.ppf(1 - alpha/2)
    z_beta   = stats.norm.ppf(power)
    mde      = (z_alpha + z_beta) * np.sqrt(2 * p * (1-p) / n)

    return {
        "metric":      metric,
        "sample_size": n,
        "baseline":    round(p * 100, 3),
        "mde":         round(mde * 100, 4),
        "mde_relative": round(mde / p * 100, 2)
    }


# ─────────────────────────────────────────
# MAIN — RUN ALL TESTS
# ─────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_clean()

    print("=" * 55)
    print("   📊 FREQUENTIST TESTS")
    print("=" * 55)

    for metric in ["retention_1", "retention_7"]:
        result = z_test(df, metric)
        sig    = "✅ SIGNIFICANT" if result["significant"] else "❌ NOT significant"
        print(f"\n  {metric.upper()}")
        print(f"  gate_30 : {result['gate_30']}%")
        print(f"  gate_40 : {result['gate_40']}%")
        print(f"  Diff    : {result['difference']:+.3f}%")
        print(f"  p-value : {result['p_value']}")
        print(f"  95% CI  : [{result['ci_low']:.3f}%, {result['ci_high']:.3f}%]")
        print(f"  Result  : {sig}")
        print(f"  Winner  : {result['winner'].upper()}")

    rounds = t_test_rounds(df)
    print(f"\n  AVG ROUNDS PLAYED")
    print(f"  gate_30 : {rounds['gate_30']}")
    print(f"  gate_40 : {rounds['gate_40']}")
    print(f"  p-value : {rounds['p_value']}")
    print(f"  Result  : {'✅ SIGNIFICANT' if rounds['significant'] else '❌ NOT significant'}")

    print("\n" + "=" * 55)
    print("   🔄 BOOTSTRAP SIMULATION")
    print("=" * 55)

    for metric in ["retention_1", "retention_7"]:
        boot = bootstrap(df, metric)
        print(f"\n  {metric.upper()}")
        print(f"  Mean diff          : {boot['mean_diff']:+.4f}%")
        print(f"  95% CI             : [{boot['ci_low']:.4f}%, {boot['ci_high']:.4f}%]")
        print(f"  P(gate_30 wins)    : {boot['prob_control_wins']}%")

    print("\n" + "=" * 55)
    print("   🎲 BAYESIAN ANALYSIS")
    print("=" * 55)

    for metric in ["retention_1", "retention_7"]:
        bayes = bayesian_analysis(df, metric)
        print(f"\n  {metric.upper()}")
        print(f"  P(gate_30 better) : {bayes['prob_gate30_better']}%")
        print(f"  P(gate_40 better) : {bayes['prob_gate40_better']}%")

    print("\n" + "=" * 55)
    print("   📏 MINIMUM DETECTABLE EFFECT")
    print("=" * 55)

    for metric in ["retention_1", "retention_7"]:
        mde = calculate_mde(df, metric)
        print(f"\n  {metric.upper()}")
        print(f"  Sample size : {mde['sample_size']:,}")
        print(f"  Baseline    : {mde['baseline']}%")
        print(f"  MDE         : {mde['mde']}% absolute")
        print(f"  MDE         : {mde['mde_relative']}% relative")