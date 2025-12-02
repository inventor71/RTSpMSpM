# RTSpMSpM Quick Reference Card

> 코드 리뷰 시 빠르게 참조할 수 있는 가이드

---

## File Quick Links

### Core Files (Must Review)

| File | Lines | Purpose |
|------|-------|---------|
| [optixSpMSpM.cpp](../optixSpMSpM/src/optixSpMSpM/optixSpMSpM.cpp) | 661 | Host code, pipeline setup |
| [optixSpMSpM.cu](../optixSpMSpM/src/optixSpMSpM/optixSpMSpM.cu) | 285 | GPU kernels |
| [optixSpMSpM.h](../optixSpMSpM/src/optixSpMSpM/optixSpMSpM.h) | 154 | Data structures |
| [Sphere.cpp](../optixSpMSpM/src/optixSpMSpM/Sphere.cpp) | 135 | Matrix B → Sphere |

### Baseline Comparison

| File | Purpose |
|------|---------|
| [cuSparse/main.cpp](../cuSparse/src/main.cpp) | cuSPARSE SpGEMM |

### Scripts

| File | Purpose |
|------|---------|
| [AE_test.py](../scripts/AE_test.py) | Benchmark runner |
| [install.sh](../scripts/install.sh) | Build script |

---

## Key Functions at a Glance

### optixSpMSpM.cpp

```
Line 73-131   : mat1ToGPU()         - Load Matrix A as rays
Line 136-162  : storeSphereData()   - Load Matrix B as spheres
Line 174-187  : contextSetUp()      - OptiX initialization
Line 189-260  : buildGAS()          - Build acceleration structure
Line 262-293  : createModule()      - Compile CUDA module
Line 295-344  : createProgramGroups()
Line 346-383  : createPipeline()
Line 385-449  : createSbt()         - Setup shader binding table
Line 452-494  : printResult()       - Output results
Line 496-660  : main()
```

### optixSpMSpM.cu

```
Line 47-76    : trace()         - Trace single ray
Line 79-94    : setPayload() / getPayload()
Line 107-202  : __raygen__rg()  - Ray generation kernel
Line 206-211  : __miss__ms()    - Miss shader (empty)
Line 214-282  : __anyhit__ch()  - ⭐ Core multiplication logic
```

---

## Data Structure Cheat Sheet

```cpp
// Matrix A element (becomes a Ray)
float3 rayData = {
    .x = row_index,      // 결과 행렬의 row
    .y = column_index,   // 공통 차원 k
    .z = value           // A[i,k] 값
};

// Matrix B element (becomes a Sphere)
float3 sphereCenter = {
    .x = row_index - 1,  // k (0-based)
    .y = col_index - 1,  // j (0-based)
    .z = 0.0f            // unused
};
float sphereRadius = 0.1f;
float sphereValue = value;  // stored separately
```

---

## Build Flags

### OptiX SpMSpM Variants

| Define | Behavior |
|--------|----------|
| (none) | ATOMIC - Normal SpMSpM with atomicAdd |
| `ARCHSUP` | Architecture support test |
| `NOMEM` | Skip memory operations |
| `NOINT` | No intersection (overhead test) |
| `NOTHING` | Empty kernel (baseline) |

### cuSPARSE Variants

| Define | Behavior |
|--------|----------|
| (none) | ALG3 - Standard SpGEMM |
| `REUSE` | ALG2 - Reuse pattern SpGEMM |

---

## Common Commands

```bash
# Build
cd /home/RTSpMSpM/optixSpMSpM/build && cmake ../src && make
cd /home/RTSpMSpM/cuSparse/src && make

# Run OptiX
./bin/optixSpMSpM -m1 "A.mtx" -m2 "B.mtx" -o "C.mtx" -l "log.txt"

# Run cuSPARSE
./cuSparse -m1 "A.mtx" -m2 "B.mtx" -o "C.mtx" -l "log.txt"

# Benchmark
python3 /home/RTSpMSpM/scripts/AE_test.py
```

---

## Code Review Focus Points

### Critical Sections

1. **Line 254-258 in optixSpMSpM.cu**
   ```cuda
   uint64_t idx = rowIndex * numCol + colIndex;
   if (idx < largestIdx) {
       atomicAdd(&(hit_data->result[idx]), resultFloat);
   }
   ```
   - Boundary check
   - Atomic operation correctness

2. **Line 118 in optixSpMSpM.cpp**
   ```cpp
   tempMatrix.push_back(make_float3(x - 1, y - 1, val));  // 0-based
   ```
   - Index conversion (1-based MTX → 0-based)

3. **Line 72 in Sphere.cpp**
   ```cpp
   point = make_float3(x - 1, y - 1, 0.f);  // 0-based conversion
   ```
   - Consistent index handling

---

## Known Issues / TODO

- `// TODO:` in code: Few items in cuSparse/main.cpp for CSR↔COO conversion
- Large matrix handling: Potential memory overflow for very large dense result buffer
- Fixed sphere radius (0.1f): May need adjustment for numerical stability

---

## Performance Metrics

Output format from AE_test.py:
```
all_result.csv    → Raw timing data
avg_result.csv    → Averaged results
final_result.csv  → Speedup comparison
```

Timing measured: "computation time no io" (excluding file I/O)

---

*Last updated: 2025-12-02*
