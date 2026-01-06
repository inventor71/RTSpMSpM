# NVTX Profiling Guide for optixSpMSpM

## Overview
The optixSpMSpM code has been instrumented with NVIDIA NVTX (NVIDIA Tools Extension) markers to enable detailed profiling with Nsight Systems (nsys).

## Prerequisites
- NVIDIA Nsight Systems installed (nsys version 2023.3.3 or later)
- CUDA Toolkit with nvToolsExt library
- Compiled optixSpMSpM binary with NVTX support

## NVTX Markers Added
The following code sections are instrumented with NVTX ranges:

1. **contextSetUp** - OptiX context initialization
2. **computation time no io** - Main computation (wraps all processing)
3. **storeSphereData** - Loading matrix 2 (sphere data)
4. **mat1ToGPU** - Loading matrix 1 to GPU
5. **buildGAS** - Building Geometry Acceleration Structure
6. **createModule** - Creating OptiX module
7. **createProgramGroups** - Creating program groups
8. **createPipeline** - Creating OptiX pipeline
9. **createSbt** - Creating Shader Binding Table
10. **optixLaunch** - Ray tracing kernel execution
11. **printResult** - Writing output matrix to file

## Profiling Commands

### Basic Profiling
```bash
cd /home/RTSpMSpM/optixSpMSpM/build

# Set the samples directory environment variable
export OPTIX_SAMPLES_SDK_DIR=/home/RTSpMSpM/optixSpMSpM/src

# Run profiling with stats output
nsys profile -o profile_output --stats=true --force-overwrite=true \
  ./bin/optixSpMSpM \
  -m1 wiki-Vote/wiki-Vote_small.mtx \
  -m2 wiki-Vote/wiki-Vote_small.mtx \
  -o result.mtx
```

### Profiling with NVTX Range Capture
Capture only specific NVTX ranges to reduce overhead:

```bash
# Capture only the computation phase
nsys profile -o profile_output \
  --capture-range=nvtx \
  --nvtx-capture="computation time no io" \
  --stats=true \
  ./bin/optixSpMSpM \
  -m1 wiki-Vote/wiki-Vote_small.mtx \
  -m2 wiki-Vote/wiki-Vote_small.mtx \
  -o result.mtx
```

### Advanced Profiling Options
```bash
# Enable CUDA API tracing and GPU trace
nsys profile -o profile_output \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --stats=true \
  ./bin/optixSpMSpM \
  -m1 wiki-Vote/wiki-Vote_small.mtx \
  -m2 wiki-Vote/wiki-Vote_small.mtx
```

## Example: wiki-Vote Profiling

### Test Case
- Matrix: wiki-Vote_small.mtx (4148×4148, 57,920 non-zeros)
- Location: `/home/RTSpMSpM/optixSpMSpM/src/data/wiki-Vote/`

### Command
```bash
export OPTIX_SAMPLES_SDK_DIR=/home/RTSpMSpM/optixSpMSpM/src
nsys profile -o wiki-Vote_profile --stats=true --force-overwrite=true \
  ./bin/optixSpMSpM \
  -m1 wiki-Vote/wiki-Vote_small.mtx \
  -m2 wiki-Vote/wiki-Vote_small.mtx \
  -o wiki-Vote_result.mtx
```

### Results Summary
**Total Execution Time:** 67.76 ms

**NVTX Timing Breakdown:**
| Function | Time (ms) | Percentage | Description |
|----------|-----------|------------|-------------|
| contextSetUp | 670.27 | 59.9% | OptiX initialization |
| printResult | 300.85 | 26.9% | Output file writing |
| computation time no io | 67.77 | 6.1% | Core computation |
| storeSphereData | 27.89 | 2.5% | Matrix 2 loading |
| mat1ToGPU | 26.00 | 2.3% | Matrix 1 loading |
| createPipeline | 9.33 | 0.8% | Pipeline creation |
| buildGAS | 2.39 | 0.2% | Acceleration structure |
| optixLaunch | 1.19 | 0.1% | Ray tracing kernel |

**GPU Kernel Statistics:**
- Primary ray generation kernel: 1.14 ms (73.7% of GPU time)
- NVIDIA internal kernels: 0.40 ms (26.3% of GPU time)

**Memory Operations:**
- Device-to-Host transfers: 28.85 ms, 68.8 MB
- Host-to-Device transfers: 0.16 ms, 1.85 MB
- Memory set operations: 0.08 ms, 68.8 MB

## Viewing Results

### Using nsys-ui (GUI)
```bash
nsys-ui profile_output.nsys-rep
```

### Extracting Reports
```bash
# Generate detailed stats
nsys stats profile_output.nsys-rep

# Export to specific formats
nsys export --type=sqlite -o output.sqlite profile_output.nsys-rep
```

## File Paths

### Environment Setup
The application expects matrix files relative to the OPTIX_SAMPLES_SDK_DIR environment variable:
- Set: `export OPTIX_SAMPLES_SDK_DIR=/home/RTSpMSpM/optixSpMSpM/src`
- Files are located under: `$OPTIX_SAMPLES_SDK_DIR/data/<dataset>/`

### Available Test Datasets
Located in `/home/RTSpMSpM/optixSpMSpM/src/data/`:
- wiki-Vote (multiple sizes: .mtx, _small.mtx, _xsmall.mtx)
- web-Google
- amazon0312
- email-Enron
- roadNet-CA
- And more...

## Integration with AE_test.py

The automated testing script `/home/RTSpMSpM/scripts/AE_test.py` can be modified to include profiling:

```python
# Example: Add profiling to automated tests
import subprocess
import os

os.environ['OPTIX_SAMPLES_SDK_DIR'] = '/home/RTSpMSpM/optixSpMSpM/src'

profile_cmd = [
    'nsys', 'profile',
    '-o', f'{dataset_name}_profile',
    '--stats=true',
    './bin/optixSpMSpM',
    '-m1', f'{dataset_name}/{dataset_name}_small.mtx',
    '-m2', f'{dataset_name}/{dataset_name}_small.mtx'
]

subprocess.run(profile_cmd, cwd='/home/RTSpMSpM/optixSpMSpM/build')
```

## Troubleshooting

### File Not Found Errors
If you see "sutil::sampleDataFilePath couldn't locate":
1. Ensure OPTIX_SAMPLES_SDK_DIR is set correctly
2. Use relative paths from the data directory (e.g., `wiki-Vote/wiki-Vote_small.mtx`)
3. Verify the file exists: `ls $OPTIX_SAMPLES_SDK_DIR/data/<your-path>`

### NVTX Markers Not Showing
1. Verify binary is linked with nvToolsExt: `ldd ./bin/optixSpMSpM | grep nvToolsExt`
2. Check nsys was run with `--trace=nvtx` (enabled by default)
3. Ensure code was compiled with NVTX headers included

### Empty GPU Traces
- GPU kernel data requires OptiX ray tracing execution
- Check that the application runs successfully without errors
- Verify CUDA is properly initialized

## Performance Optimization Tips

Based on profiling results:
1. **Context initialization (60%)** - One-time cost, can be cached for multiple runs
2. **File I/O (27%)** - Consider binary formats or streaming for large outputs
3. **GPU computation (6%)** - The actual SpMSpM work is already efficient
4. **Data loading (5%)** - Consider batching or async loading for multiple matrices

## Additional Resources
- [NVTX Documentation](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvtx)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [OptiX Programming Guide](https://raytracing-docs.nvidia.com/optix7/guide/index.html)
