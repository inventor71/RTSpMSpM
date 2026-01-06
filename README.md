# RTSpMSpM: Harnessing Ray Tracing for Efficient Sparse Matrix Computations

This repository contains the code and benchmark suite for **RTSpMSpM**, a novel approach that leverages NVIDIA’s hardware-accelerated ray tracing (RT Cores) to speed up **Sparse Matrix × Sparse Matrix Multiplication (SpMSpM)**. This project demonstrates the feasibility and benefits of mapping sparse matrix operations to the ray tracing pipeline.


## Technologies Used

- **Languages**: C++, Python
- **GPU Frameworks**: CUDA 12.3, NVIDIA OptiX 8.0.0, cuSPARSE
- **Build Tools**: CMake 3.22, GCC 7.5.0
- **Containers**: Docker 27.3.1 with NVIDIA support
- **Datasets**: SuiteSparse Matrix Collection

## Project Structure

```
RTSpMSpM/
├── cuSparse/                # GPU baseline using cuSPARSE
├── Dockerfile/              # Docker build scripts
├── optixSpMSpM/             # OptiX SDK and build system
│   ├── build/               # Compiled binaries and CMake output
│   └── src/            
│       ├── data/            # Input Datasets
│       ├── support/
│       ├── sutil/     
│       └── optixSpMSpM/     # Core ray tracing-based SpMSpM logic
└── Tool/
    └── Script/
        ├── AE_test.py               # Main script to launch experiments and benchmark
        ├── install.sh               # Compile program
        └── download_dataset.sh      # Benchmark automation script
```


## Installation and Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/escalab/RTSpMSpM.git
cd RTSpMSpM
```

### Step 2: Build the Docker Image
```bash
cd Dockerfile
./build_image.sh
```

### Step 3: Start the Docker Container
```bash
./start_image.sh
```

### Step 4: Enter the Docker Container
```bash
./run.sh
```

### Step 5: Install and Compile Inside the Container
```bash
cd RTSpMSpM/scripts
./install.sh
```


## How to Run

If runned the installed script, skip to step 3: To run the experiment

### To compile the GPU Baseline:
```bash
cd /home/RTSpMSpM/cuSparse/src
make
```

### To compile the RT-based SpMSpM implementation:
```bash
mkdir -p /home/RTSpMSpM/optixSpMSpM/build
cd /home/RTSpMSpM/optixSpMSpM/build
cmake ../src
make
```

### To run the experiment:
```bash
python3 /home/RTSpMSpM/scripts/AE_test.py
```


## Profiling with NVTX and Nsight Systems

The RTSpMSpM code is instrumented with NVIDIA NVTX (Tools Extension) markers for detailed performance profiling.

### Install NVIDIA Nsight Systems

Install nsys for CUDA 12.3 (match your CUDA version):

```bash
apt-get update
apt-get install -y cuda-nsight-systems-12-3
```

Verify installation:
```bash
nsys --version
# Expected output: NVIDIA Nsight Systems version 2023.3.3.42 (or similar)
```

### NVTX Markers

The following code sections are instrumented with NVTX profiling ranges:
- `contextSetUp` - OptiX initialization
- `computation time no io` - Main computation phase
- `storeSphereData` - Matrix 2 loading
- `mat1ToGPU` - Matrix 1 loading to GPU
- `buildGAS` - Geometry Acceleration Structure build
- `createModule`, `createProgramGroups`, `createPipeline`, `createSbt` - OptiX setup
- `optixLaunch` - Ray tracing kernel execution
- `printResult` - Output file writing

### Running Profiling

Set the samples directory environment variable:
```bash
export OPTIX_SAMPLES_SDK_DIR=/home/RTSpMSpM/optixSpMSpM/src
```

Basic profiling command:
```bash
cd /home/RTSpMSpM/optixSpMSpM/build

nsys profile -o profile_output --stats=true \
  ./bin/optixSpMSpM \
  -m1 wiki-Vote/wiki-Vote_small.mtx \
  -m2 wiki-Vote/wiki-Vote_small.mtx \
  -o result.mtx
```

Profile only computation phase (reduced overhead):
```bash
nsys profile -o profile_output \
  --capture-range=nvtx \
  --nvtx-capture="computation time no io" \
  --stats=true \
  ./bin/optixSpMSpM \
  -m1 wiki-Vote/wiki-Vote_small.mtx \
  -m2 wiki-Vote/wiki-Vote_small.mtx
```

### View Results

Using the GUI (requires X11 forwarding or local install):
```bash
nsys-ui profile_output.nsys-rep
```

Or generate text reports:
```bash
nsys stats profile_output.nsys-rep
```

### Example Output

#### RTSpMSpM (OptiX) - wiki-Vote_small.mtx (4148×4148, 57,920 non-zeros):

| Function | Time (ms) | % |
|----------|-----------|---|
| contextSetUp | 670.3 | 59.9% |
| printResult | 300.9 | 26.9% |
| computation time no io | 67.8 | 6.1% |
| storeSphereData | 27.9 | 2.5% |
| mat1ToGPU | 26.0 | 2.3% |
| optixLaunch (GPU kernel) | 1.2 | 0.1% |

#### cuSparse Baseline - wiki-Vote_small.mtx (4148×4148, 57,920 non-zeros):

| Function | Time (ms) | % |
|----------|-----------|---|
| computation time no io | 452.2 | 47.6% |
| printCooToFile | 264.0 | 27.8% |
| loadMatrices | 180.4 | 19.0% |
| spgemm_kernel | 47.6 | 5.0% |
| copyResultsToHost | 4.6 | 0.5% |
| coo2csr | 0.14 | 0.0% |
| csr2coo | 0.12 | 0.0% |

### Profiling cuSparse Baseline

The cuSparse baseline also includes NVTX profiling markers:

```bash
cd /home/RTSpMSpM/cuSparse/src

nsys profile -o cuSparse_profile --stats=true \
  ./cuSparse \
  -m1 /home/RTSpMSpM/optixSpMSpM/src/data/wiki-Vote/wiki-Vote_small.mtx \
  -m2 /home/RTSpMSpM/optixSpMSpM/src/data/wiki-Vote/wiki-Vote_small.mtx \
  -o result.mtx
```

**For complete profiling documentation, see:** `/home/RTSpMSpM/NVTX_PROFILING_GUIDE.md`


## 6. Artifact Details

- **Artifact Availability**: Public  
  [📦 Zenodo Archive](https://zenodo.org/record/8210452)  
  [💻 GitHub Repo](https://github.com/escalab/RTSpMSpM)

- **Expected Output**:  
  Execution time (latency in milliseconds) for sparse matrix benchmarks, shown in logs or console output.

- **Evaluation Time**:
  - Setup: ~10 minutes
  - Experiment Runtime: 2–3 hours

- **Hardware Requirements**:
  - GPU: NVIDIA GPU with compute capability 5.0+ (7.5 recommended)
  - CPU: Original Hardware Intel Core i7 14700K
  - RAM: Original Hardware 128GB DDR4
  - Disk Space: up to ~2GB per datasets

- **Software Requirements**:
  - CUDA 12.3
  - Docker 27.3.1 (or cmake 3.22 + gcc 7.5.0 if building natively)
  - nvidia-docker recommended for easy setup

- **Licensing**:
  - Code: MIT License
  - Datasets: Original SuiteSparse licenses

---

## 7. Citation

If you use this artifact in your research, please cite the corresponding ISCA 2025 paper:

> *RT+SpMSpM: Harnessing Ray Tracing for Efficient Sparse Matrix Computations*, ISCA 2025.
