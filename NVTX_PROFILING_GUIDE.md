# NVTX Profiling Guide for RTSpMSpM

## Overview
Both the **RTSpMSpM (OptiX-based)** implementation and the **cuSparse baseline** have been instrumented with NVIDIA NVTX (NVIDIA Tools Extension) markers to enable detailed profiling with Nsight Systems (nsys). This allows for fine-grained performance analysis and comparison between the two approaches.

## Prerequisites
- NVIDIA Nsight Systems installed (nsys version 2023.3.3 or later)
- CUDA Toolkit 12.3 with nvToolsExt library
- Compiled binaries with NVTX support (both already built with NVTX)

## Installation

If nsys is not already installed:

```bash
apt-get update
apt-get install -y cuda-nsight-systems-12-3
```

Verify installation:
```bash
nsys --version
# Expected: NVIDIA Nsight Systems version 2023.3.3.42 or similar
```

## NVTX Markers Implemented

### RTSpMSpM (OptiX) - `/home/RTSpMSpM/optixSpMSpM/src/optixSpMSpM/optixSpMSpM.cpp`

1. **contextSetUp** - OptiX context initialization
2. **computation time no io** - Main computation phase (wraps all GPU processing)
3. **storeSphereData** - Loading matrix 2 (sphere data structure)
4. **mat1ToGPU** - Loading matrix 1 to GPU memory
5. **buildGAS** - Building Geometry Acceleration Structure
6. **createModule** - Creating OptiX module
7. **createProgramGroups** - Creating program groups
8. **createPipeline** - Creating OptiX pipeline
9. **createSbt** - Creating Shader Binding Table
10. **optixLaunch** - Ray tracing kernel execution (actual SpMSpM computation)
11. **printResult** - Writing output matrix to file

### cuSparse Baseline - `/home/RTSpMSpM/cuSparse/src/main.cpp`

Both `compute()` and `reuseCompute()` functions are instrumented:

1. **loadMatrices** - Loading input matrices from files
2. **computation time no io** - Main computation phase (wraps all GPU processing)
3. **coo2csr** - Converting COO format to CSR format
4. **spgemm_kernel** (in `compute`) or **spgemm_reuse** (in `reuseCompute`) - cuSparse SpGEMM computation
5. **csr2coo** - Converting CSR format back to COO format for output
6. **copyResultsToHost** - Copying results from device to host memory
7. **printCooToFile** - Writing output matrix to file

## Profiling RTSpMSpM (OptiX)

### Basic Profiling

```bash
cd /home/RTSpMSpM/optixSpMSpM/build

# Set environment variable for data path
export OPTIX_SAMPLES_SDK_DIR=/home/RTSpMSpM/optixSpMSpM/src

# Run profiling with stats
nsys profile -o rtspmspm_profile --stats=true \
  ./bin/optixSpMSpM \
  -m1 wiki-Vote/wiki-Vote_small.mtx \
  -m2 wiki-Vote/wiki-Vote_small.mtx \
  -o result.mtx
```

### Capture Specific NVTX Range

```bash
# Profile only the computation phase (reduces overhead)
nsys profile -o rtspmspm_profile \
  --capture-range=nvtx \
  --nvtx-capture="computation time no io" \
  --stats=true \
  ./bin/optixSpMSpM \
  -m1 wiki-Vote/wiki-Vote_small.mtx \
  -m2 wiki-Vote/wiki-Vote_small.mtx
```

### Example Results: wiki-Vote_small.mtx (4148×4148, 57,920 non-zeros)

| Function | Time (ms) | Percentage | Description |
|----------|-----------|------------|-------------|
| contextSetUp | 670.3 | 59.9% | OptiX initialization overhead |
| printResult | 300.9 | 26.9% | File I/O |
| **computation time no io** | **67.8** | **6.1%** | **Core computation** |
| storeSphereData | 27.9 | 2.5% | Matrix 2 loading |
| mat1ToGPU | 26.0 | 2.3% | Matrix 1 loading |
| optixLaunch | 1.2 | 0.1% | GPU kernel execution |

**Key Insight:** The actual ray tracing computation (optixLaunch) takes only 1.2ms, while initialization dominates the runtime.

## Profiling cuSparse Baseline

### Basic Profiling

```bash
cd /home/RTSpMSpM/cuSparse/src

# Run profiling with stats
nsys profile -o cusparse_profile --stats=true \
  ./cuSparse \
  -m1 /home/RTSpMSpM/optixSpMSpM/src/data/wiki-Vote/wiki-Vote_small.mtx \
  -m2 /home/RTSpMSpM/optixSpMSpM/src/data/wiki-Vote/wiki-Vote_small.mtx \
  -o result.mtx
```

### Capture Computation Phase

```bash
# Profile only the computation (no I/O)
nsys profile -o cusparse_profile \
  --capture-range=nvtx \
  --nvtx-capture="computation time no io" \
  --stats=true \
  ./cuSparse \
  -m1 /home/RTSpMSpM/optixSpMSpM/src/data/wiki-Vote/wiki-Vote_small.mtx \
  -m2 /home/RTSpMSpM/optixSpMSpM/src/data/wiki-Vote/wiki-Vote_small.mtx
```

### Example Results: wiki-Vote_small.mtx (4148×4148, 57,920 non-zeros)

| Function | Time (ms) | Percentage | Description |
|----------|-----------|------------|-------------|
| **computation time no io** | **452.2** | **47.6%** | **Total GPU processing** |
| printCooToFile | 264.0 | 27.8% | File I/O |
| loadMatrices | 180.4 | 19.0% | Loading input matrices |
| spgemm_kernel | 47.6 | 5.0% | SpGEMM kernel execution |
| copyResultsToHost | 4.6 | 0.5% | Result transfer |
| coo2csr | 0.14 | 0.0% | Format conversion |
| csr2coo | 0.12 | 0.0% | Format conversion |

**Key Insight:** The cuSparse SpGEMM kernel takes 47.6ms compared to RTSpMSpM's 1.2ms for the same matrix.

## Comparative Analysis

### Computation Time Comparison (wiki-Vote_small.mtx)

| Metric | RTSpMSpM (OptiX) | cuSparse | Speedup |
|--------|------------------|----------|---------|
| Kernel execution | 1.2 ms | 47.6 ms | **39.7×** |
| Total computation (no I/O) | 67.8 ms | 452.2 ms | **6.7×** |
| Including initialization | 1119.2 ms | 949.0 ms | 0.85× |

**Analysis:**
- RTSpMSpM shows significant speedup in pure computation
- Initialization overhead (OptiX context setup) is amortized over multiple runs
- cuSparse has lower startup overhead but slower kernel execution

## Viewing Profiling Results

### Using nsys-ui (GUI)

Requires X11 forwarding or local nsys-ui installation:

```bash
nsys-ui profile_output.nsys-rep
```

This opens the Nsight Systems GUI showing:
- Timeline visualization with NVTX ranges
- GPU kernel execution details
- Memory transfer patterns
- CPU/GPU utilization

### Command-Line Stats

```bash
# Generate detailed statistics
nsys stats profile_output.nsys-rep

# Filter specific stats
nsys stats profile_output.nsys-rep --report nvtx_sum
nsys stats profile_output.nsys-rep --report cuda_gpu_kern_sum
```

### Export to SQLite

```bash
# Export for custom analysis
nsys export --type=sqlite -o output.sqlite profile_output.nsys-rep

# Query with sqlite3
sqlite3 output.sqlite "SELECT * FROM NVTX_EVENTS;"
```

## Advanced Profiling Options

### Full Trace with Memory Analysis

```bash
nsys profile -o detailed_profile \
  --trace=cuda,nvtx,osrt,cublas,cusparse \
  --cuda-memory-usage=true \
  --gpu-metrics-device=0 \
  --stats=true \
  ./bin/optixSpMSpM -m1 ... -m2 ...
```

### Sampling CPU Performance

```bash
nsys profile -o cpu_profile \
  --sample=cpu \
  --backtrace=fp \
  --stats=true \
  ./cuSparse -m1 ... -m2 ...
```

### Profile Multiple Runs

```bash
# Profile 5 iterations to see warm-up effects
for i in {1..5}; do
  nsys profile -o run_${i}_profile --stats=true \
    ./bin/optixSpMSpM -m1 wiki-Vote/wiki-Vote_small.mtx -m2 wiki-Vote/wiki-Vote_small.mtx
done
```

## Integration with Automated Testing

### Modify AE_test.py for Profiling

```python
import subprocess
import os

def profile_test(binary, matrix_file, output_name):
    """Add profiling to test runs"""
    env = os.environ.copy()
    if 'optixSpMSpM' in binary:
        env['OPTIX_SAMPLES_SDK_DIR'] = '/home/RTSpMSpM/optixSpMSpM/src'

    profile_cmd = [
        'nsys', 'profile',
        '-o', f'profiles/{output_name}',
        '--stats=true',
        '--force-overwrite=true',
        binary,
        '-m1', matrix_file,
        '-m2', matrix_file
    ]

    subprocess.run(profile_cmd, env=env, capture_output=True)
```

## Troubleshooting

### No NVTX Ranges Showing

1. Verify NVTX symbols are in binary:
   ```bash
   nm ./bin/optixSpMSpM | grep nvtx
   nm ./cuSparse | grep nvtx
   ```

2. Check nsys is capturing NVTX:
   ```bash
   # Should show NVTX in trace types
   nsys profile --help | grep nvtx
   ```

3. Ensure `--trace=nvtx` is enabled (it's on by default)

### Environment Variable Issues (optixSpMSpM)

If you see "sutil::sampleDataFilePath couldn't locate":
```bash
# Must set this before running
export OPTIX_SAMPLES_SDK_DIR=/home/RTSpMSpM/optixSpMSpM/src

# Verify it's set
echo $OPTIX_SAMPLES_SDK_DIR
```

### Large Profile Files

Profile files can be large for complex runs:
```bash
# Reduce overhead by capturing specific ranges
--capture-range=nvtx --nvtx-capture="computation time no io"

# Limit trace duration
--duration=10  # Profile for 10 seconds

# Sample instead of full trace
--sample=cpu --cuda-graph-trace=node
```

## Performance Optimization Insights

### From RTSpMSpM Profiling

1. **Initialization dominates single-run performance** (60%)
   - Solution: Amortize over multiple matrices
   - Keep OptiX context alive for batch processing

2. **File I/O is significant** (27%)
   - Solution: Use binary formats or in-memory data structures
   - Pipeline I/O with computation

3. **Actual computation is very fast** (6%)
   - Ray tracing kernel: 1.2ms for 57K non-zeros
   - Shows efficiency of RT Cores for SpMSpM

### From cuSparse Profiling

1. **SpGEMM kernel time** (5%)
   - 47.6ms for the same workload
   - Could benefit from tuning or different algorithms

2. **Format conversions are cheap** (< 0.3ms combined)
   - COO ↔ CSR conversions add minimal overhead

3. **Computation dominates** (48%)
   - Less initialization overhead than OptiX
   - But slower kernel execution

## Available Test Datasets

Located in `/home/RTSpMSpM/optixSpMSpM/src/data/`:

| Dataset | Dimensions | Non-zeros | File |
|---------|------------|-----------|------|
| wiki-Vote | 4148×4148 | 57,920 | wiki-Vote_small.mtx |
| email-Enron | Varies | Varies | email-Enron_small.mtx |
| web-Google | Varies | Varies | web-Google_small.mtx |
| amazon0312 | Varies | Varies | amazon0312_small.mtx |
| roadNet-CA | Varies | Varies | roadNet-CA_small.mtx |

Use `_small.mtx` variants for faster profiling iterations.

## Additional Resources

- [NVTX Documentation](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvtx)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [OptiX Programming Guide](https://raytracing-docs.nvidia.com/optix7/guide/index.html)
- [cuSPARSE Library Documentation](https://docs.nvidia.com/cuda/cusparse/)

## Quick Reference Commands

```bash
# Profile RTSpMSpM
cd /home/RTSpMSpM/optixSpMSpM/build
export OPTIX_SAMPLES_SDK_DIR=/home/RTSpMSpM/optixSpMSpM/src
nsys profile -o profile --stats=true ./bin/optixSpMSpM -m1 wiki-Vote/wiki-Vote_small.mtx -m2 wiki-Vote/wiki-Vote_small.mtx

# Profile cuSparse
cd /home/RTSpMSpM/cuSparse/src
nsys profile -o profile --stats=true ./cuSparse \
  -m1 /home/RTSpMSpM/optixSpMSpM/src/data/wiki-Vote/wiki-Vote_small.mtx \
  -m2 /home/RTSpMSpM/optixSpMSpM/src/data/wiki-Vote/wiki-Vote_small.mtx

# View results
nsys-ui profile.nsys-rep  # GUI
nsys stats profile.nsys-rep --report nvtx_sum  # Command line
```
