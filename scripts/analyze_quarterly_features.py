import pandas as pd
import yaml
import os
import numpy as np

# Paths
ROOT = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(ROOT, 'config.yaml')
processed_dir = os.path.join(ROOT, 'data', 'processed')
file_path = os.path.join(processed_dir, 'ACB_features.csv')

# Load config
with open(config_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

fundamental = cfg.get('features', {}).get('fundamental', [])
banking = cfg.get('features', {}).get('banking_specific', [])
timesteps = cfg.get('training', {}).get('timesteps', 30)

print(f'Loaded config: timesteps={timesteps}')
print(f'Fundamental cols: {fundamental}')
print(f'Banking-specific cols: {banking}')

# Load dataframe
if not os.path.exists(file_path):
    print(f'File not found: {file_path}')
    raise SystemExit(1)

df = pd.read_csv(file_path)
print(f'ACB features shape: {df.shape}')

cols_to_check = fundamental + banking

results = []

for col in cols_to_check:
    info = {'col': col}
    if col not in df.columns:
        info.update({'present': False})
        results.append(info)
        continue
    s = df[col]
    info['present'] = True
    info['n_rows_nonnull'] = int(s.notna().sum())
    info['n_unique'] = int(s.nunique(dropna=True))
    info['pct_null'] = float(s.isna().mean())
    # how often it changes day-to-day
    shifted = s.fillna(method='ffill')
    changes = (shifted != shifted.shift(1))
    info['pct_days_changed'] = float(changes.mean())
    # sliding windows constancy
    L = len(s.dropna())
    total_windows = max(0, len(df) - timesteps + 1)
    if total_windows > 0:
        const_windows = 0
        for i in range(0, len(df) - timesteps + 1):
            window = df[col].iloc[i:i+timesteps]
            # consider window constant if after filling na it's all same
            w = window.dropna()
            if len(w) == 0:
                # treat as constant (no info)
                const_windows += 1
            elif w.nunique() == 1:
                const_windows += 1
        info['total_windows'] = total_windows
        info['const_windows'] = const_windows
        info['pct_windows_constant'] = const_windows / total_windows
    else:
        info['total_windows'] = 0
        info['const_windows'] = None
        info['pct_windows_constant'] = None

    results.append(info)

# Print summary
print('\nColumn summary:')
for r in results:
    if not r['present']:
        print(f"- {r['col']}: MISSING")
        continue
    print(f"- {r['col']}: unique={r['n_unique']}, pct_null={r['pct_null']:.3f}, pct_days_changed={r['pct_days_changed']:.3f}, pct_windows_constant={r['pct_windows_constant']:.3f} (windows={r['total_windows']})")

# Quick decision guidance
print('\nGuidance:')
for r in results:
    if not r.get('present', False):
        continue
    pct_const = r.get('pct_windows_constant', 0)
    if pct_const is None:
        verdict = 'N/A'
    elif pct_const > 0.9:
        verdict = 'VERY LOW VARIABILITY -> likely uninformative at this timestep'
    elif pct_const > 0.5:
        verdict = 'LOW VARIABILITY -> possibly weak signal'
    else:
        verdict = 'VARIES -> can provide signal'
    print(f"{r['col']}: {verdict}")

# Save results to csv
out_path = os.path.join(processed_dir, 'ACB_quarterly_feature_analysis.csv')
pd.DataFrame(results).to_csv(out_path, index=False)
print(f'Wrote detailed results to {out_path}')
