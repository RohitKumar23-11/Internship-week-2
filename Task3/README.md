
README.md

# 📊 Time-Series Method Benchmarking

This project benchmarks different implementations of time-series operations  
(e.g., rolling mean, rolling variance) using Pandas, NumPy, and Numba.

It measures performance across datasets of varying sizes and produces:
- A CSV file (`results.csv`) with benchmark results
- A plot (`benchmark_plot.png`) comparing execution times on a log-log scale

## 📂 Project Structure


.
├── benchmark.py        # Main benchmarking script
├── timeseries\_utils.py # Utility functions for time-series calculations
├── results.csv         # Generated benchmark results (after running)
├── benchmark\_plot.png  # Generated plot (after running)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation


## 🚀 Usage
1. Install dependencies  
   pip install -r requirements.txt


2. Run the benchmark
   python benchmark.py
 

3. View results

   * `results.csv` for raw data
   * `benchmark_plot.png` for performance visualization




