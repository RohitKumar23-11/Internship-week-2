# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 11:56:21 2025

@author: Acer
"""

import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from timeseries_utils import *

sizes = [10_000, 100_000, 1_000_000, 5_000_000]
window = 100
results = []

for size in sizes:
    data = np.random.randn(size)
    df = pd.Series(data)
    
    # Rolling Mean Benchmarks
    for method_name, func in [
            ('pandas', lambda: rolling_mean_pd(df, window)),
            ('numpy', lambda: rolling_mean_np(data, window)),
            ('numba', lambda: rolling_mean_numba(data, window))
            ]:
        try:
            start = time.time()
            func()
            duration = time.time() - start
            results.append([size, 'rolling_mean', method_name, duration])
        except Exception  as e:
            results.append([size, 'rolling_mean', method_name, -1])
            
# save results
benchmark_df = pd.DataFrame(results, columns=['data_size', 'operation', 'method', 'time_sec'])
benchmark_df.to_csv('results.csv', index=False)

#Plotting
df = pd.read_csv('results.csv')
df = df[df['time_sec'] > 0 ]

print(df.groupby(['method']).size())

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x = 'data_size', y='time_sec', hue='method', style='operation', markers=True, dashes=True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Data Size (log scale)')
plt.ylabel('Execution Time (s, log scale)')
plt.title('Benchmark of time-series methods')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('benchmark_plot.png', dpi=300)
plt.show()