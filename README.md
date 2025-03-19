# Installation & Setup
Download the necessary repositories
```
# Clone the main repository
git clone https://github.com/dkoh555/ce452_final_project.git

# Navigate into the repository
cd ce452_final_project

# Initialize and fetch submodules
git submodule init
git submodule update
```
Install Genesis locally for development
```
# Navigate into the Genesis_optimized submodule
cd Genesis_optimized

# Install in development mode
pip install -e .

# Return to main repository
cd ..
```

# Running the Benchmark Code
Simply run parallel_benchmarking.py

# Switching between Optimized Versions
By default, you are on the latest Optimization V2 code.
### For OG code
```
git checkout f28fa1743dc1776f15a848764772330d6b8ec5a0 && git submodule update
```
Then run parallel_benchmarking.py

### For Optimization V1 code
```
git checkout ccb61e876dfe4279f1833a2ace259cd48c63f4e3 && git submodule update
```
Then run parallel_benchmarking.py

### For Optimization V2 code (Latest)
```
git checkout main && git submodule update
```
Then run parallel_benchmarking.py
