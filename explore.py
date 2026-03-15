import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/cookie_cats.csv")

print("=" * 50)
print("   🎮 COOKIE CATS — DATA EXPLORATION")
print("=" * 50)

# Basic dataset information
print(f"\n📋 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n📋 Columns:\n{df.dtypes}")
print(f"\n📋 First 5 rows:\n{df.head()}")
print(f"\n📋 Missing values:\n{df.isnull().sum()}")
print(f"\n📋 Group sizes:\n{df['version'].value_counts()}")

# Compute key metrics for each A/B group
print("\n" + "=" * 50)
print("   📊 KEY METRICS BY GROUP")
print("=" * 50)

for group in ["gate_30", "gate_40"]:
    subset = df[df["version"] == group]
    r1     = subset["retention_1"].mean() * 100
    r7     = subset["retention_7"].mean() * 100
    rounds = subset["sum_gamerounds"].mean()

    print(f"\n  {group.upper()}")
    print(f"  Users         : {len(subset):,}")
    print(f"  1-day  retention : {r1:.2f}%")
    print(f"  7-day  retention : {r7:.2f}%")
    print(f"  Avg rounds played: {rounds:.1f}")

# Data quality checks
print("\n" + "=" * 50)
print("   🔍 DATA QUALITY CHECKS")
print("=" * 50)

# Detect duplicate user IDs
dupes = df[df.duplicated("userid", keep=False)]
print(f"\n  Duplicate user IDs : {len(dupes)}")

# Identify users who played zero rounds
zero_rounds = df[df["sum_gamerounds"] == 0]
print(f"  Users with 0 rounds: {len(zero_rounds)}")

# Detect extreme outliers using the 99th percentile
q99 = df["sum_gamerounds"].quantile(0.99)
outliers = df[df["sum_gamerounds"] > q99]

print(f"  99th percentile    : {q99:.0f} rounds")
print(f"  Outliers above p99 : {len(outliers)}")
print(f"\n  Max rounds played  : {df['sum_gamerounds'].max():,}")