# RTSpMSpM Code Review Guide

> **RT+SpMSpM: Harnessing Ray Tracing for Efficient Sparse Matrix Computations** (ISCA 2025)

이 문서는 RTSpMSpM 코드베이스를 처음 접하는 개발자가 코드 리뷰를 효율적으로 수행할 수 있도록 작성되었습니다.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Directory Structure](#directory-structure)
4. [Core Components](#core-components)
5. [Data Flow](#data-flow)
6. [Key Algorithms](#key-algorithms)
7. [Build & Run](#build--run)
8. [Code Review Checklist](#code-review-checklist)

---

## Project Overview

### 핵심 아이디어
RTSpMSpM은 NVIDIA의 **RT Cores (Ray Tracing 하드웨어)**를 활용하여 **Sparse Matrix × Sparse Matrix Multiplication (SpMSpM)**을 가속화하는 혁신적인 접근법입니다.

### 핵심 매핑 전략
| Sparse Matrix 개념 | Ray Tracing 개념 |
|-------------------|-----------------|
| Matrix A의 non-zero 원소 | **Ray (광선)** |
| Matrix B의 non-zero 원소 | **Sphere (구체)** |
| 곱셈이 가능한 원소 쌍 (A[i,k] × B[k,j]) | **Ray-Sphere Intersection** |

```
Matrix A (m×k)  ×  Matrix B (k×n)  =  Matrix C (m×n)

  A[i,k]  →  Ray with origin at column k
  B[k,j]  →  Sphere at position (k, j)

  Intersection  →  C[i,j] += A[i,k] × B[k,j]
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        RTSpMSpM System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   Input     │     │   OptiX     │     │   Output    │       │
│  │  (.mtx)     │────▶│  Pipeline   │────▶│  (.mtx)     │       │
│  │  Matrix A,B │     │             │     │  Matrix C   │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────┐      │
│  │               NVIDIA OptiX Pipeline                   │      │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │      │
│  │  │  Ray    │  │   GAS   │  │   RT    │  │  Any    │  │      │
│  │  │  Gen    │──│  Build  │──│  Core   │──│  Hit    │  │      │
│  │  │         │  │         │  │ Traverse│  │         │  │      │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
RTSpMSpM/
├── 📁 optixSpMSpM/           # 🔥 핵심 RT 기반 SpMSpM 구현
│   ├── 📁 include/           # OptiX SDK 헤더
│   └── 📁 src/
│       ├── 📁 optixSpMSpM/   # ⭐ 메인 구현 (최우선 리뷰 대상)
│       ├── 📁 sutil/         # 유틸리티 라이브러리
│       ├── 📁 cuda/          # CUDA 헬퍼 코드
│       └── 📁 support/       # 서드파티 라이브러리 (GLFW, imgui 등)
│
├── 📁 cuSparse/              # 🔄 GPU 베이스라인 (cuSPARSE 구현)
│   └── 📁 src/
│
├── 📁 scripts/               # 🧪 실험 및 벤치마크 스크립트
│
└── 📁 Dockerfile/            # 🐳 Docker 환경 설정
```

---

## Core Components

### 1. optixSpMSpM (메인 구현) ⭐

| 파일 | 설명 | 우선순위 |
|-----|------|---------|
| [`optixSpMSpM.cpp`](../optixSpMSpM/src/optixSpMSpM/optixSpMSpM.cpp) | 메인 호스트 코드, OptiX 파이프라인 설정 | 🔴 High |
| [`optixSpMSpM.cu`](../optixSpMSpM/src/optixSpMSpM/optixSpMSpM.cu) | GPU 커널: Ray Generation, Any-Hit, Miss | 🔴 High |
| [`optixSpMSpM.h`](../optixSpMSpM/src/optixSpMSpM/optixSpMSpM.h) | 데이터 구조체 정의 | 🔴 High |
| [`Sphere.cpp`](../optixSpMSpM/src/optixSpMSpM/Sphere.cpp) | Matrix B를 Sphere로 변환 | 🟡 Medium |
| [`Sphere.h`](../optixSpMSpM/src/optixSpMSpM/Sphere.h) | Sphere 클래스 정의 | 🟡 Medium |

#### 1.1 optixSpMSpM.cpp - 호스트 코드 (661 lines)

**핵심 함수 분석:**

```cpp
// 실행 흐름
main()
├── storeSphereData()     // Matrix B → Sphere 변환 (Line 136-162)
├── mat1ToGPU()           // Matrix A → Ray 데이터 변환 (Line 73-131)
├── contextSetUp()        // OptiX 초기화 (Line 174-187)
├── buildGAS()            // Geometry Acceleration Structure 빌드 (Line 189-260)
├── createModule()        // CUDA 모듈 컴파일 (Line 262-293)
├── createProgramGroups() // 프로그램 그룹 생성 (Line 295-344)
├── createPipeline()      // 파이프라인 링킹 (Line 346-383)
├── createSbt()           // Shader Binding Table 설정 (Line 385-449)
├── optixLaunch()         // 🚀 RT Core 실행 (Line 608)
└── printResult()         // 결과 출력 (Line 452-494)
```

**주요 함수 상세:**

| 함수 | 라인 | 역할 |
|-----|------|-----|
| `mat1ToGPU()` | 73-131 | MTX 파일에서 Matrix A를 읽어 `float3(row, col, value)` 형태로 GPU 메모리에 로드 |
| `storeSphereData()` | 136-162 | Matrix B를 Sphere 객체로 변환, 각 non-zero 원소가 3D 공간의 구체가 됨 |
| `buildGAS()` | 189-260 | Sphere들을 OptiX의 Geometry Acceleration Structure로 빌드 (BVH 트리) |
| `createSbt()` | 385-449 | Ray/Sphere 데이터를 Shader Binding Table에 바인딩 |

#### 1.2 optixSpMSpM.cu - GPU 커널 (285 lines)

**핵심 커널:**

```cuda
// Ray Generation (Line 113-202)
__raygen__rg()
├── 각 Ray(Matrix A의 원소)에 대해 실행
├── Ray의 origin = (column_index, 0, 0)
├── Ray의 direction = (column_index, 1e16, 0)
└── trace() 호출로 RT Core 탐색 시작

// Any-Hit (Line 220-282) - ⭐ 핵심 곱셈 로직
__anyhit__ch()
├── Ray-Sphere 충돌 시 호출
├── A[i,k] × B[k,j] 곱셈 수행
├── atomicAdd()로 C[i,j]에 결과 누적
└── optixIgnoreIntersection()으로 계속 탐색

// Miss (Line 206-211)
__miss__ms()
└── 충돌 없을 때 호출 (빈 구현)
```

**빌드 옵션 (전처리기 매크로):**

| 매크로 | 설명 |
|-------|------|
| `ATOMIC` (기본) | 정상적인 SpMSpM 연산, atomicAdd 사용 |
| `ARCHSUP` | 아키텍처 지원 테스트 |
| `NOMEM` | 메모리 연산 제외 (성능 분석용) |
| `NOINT` | 인터섹션 없는 버전 (오버헤드 측정용) |
| `NOTHING` | 빈 커널 (베이스라인 측정용) |

#### 1.3 optixSpMSpM.h - 데이터 구조체 (154 lines)

```cpp
// Ray 데이터 (Matrix A 원소)
struct RayData {
    float3* originVec;  // (row, col, value)
    uint64_t size;      // non-zero 개수
};

// Sphere 데이터 (Matrix B 원소)
struct SphereData {
    float* sphereColor;    // Matrix B의 values
    float* result;         // 결과 Matrix C
    int resultNumRow;
    int resultNumCol;
    uint64_t matrix1size;
    uint64_t matrix2size;
};

// OptiX 상태 관리
struct optixState {
    float3* d_matrix;              // GPU상의 Matrix A
    float* spherePoints;           // GPU상의 Matrix B values
    float* d_result;               // GPU상의 결과 버퍼
    OptixTraversableHandle gas_handle;  // GAS 핸들
    // ... OptiX 파이프라인 컴포넌트들
};
```

### 2. cuSparse (베이스라인)

| 파일 | 설명 |
|-----|------|
| [`main.cpp`](../cuSparse/src/main.cpp) | cuSPARSE를 사용한 SpGEMM 구현 |
| [`util.cpp`](../cuSparse/src/util.cpp) | 유틸리티 함수 (MTX 파싱 등) |
| [`Timing.cpp`](../cuSparse/src/Timing.cpp) | 성능 측정 |

**핵심 함수:**

```cpp
// main.cpp
compute()           // SpGEMM 알고리즘 3 사용 (Line 365-591)
reuseCompute()      // SpGEMM 알고리즘 2 (재사용 최적화) (Line 98-363)

// 주요 cuSPARSE API 호출
cusparseSpGEMM_workEstimation()
cusparseSpGEMM_compute()
cusparseSpGEMM_copy()
```

### 3. 스크립트

| 파일 | 설명 |
|-----|------|
| [`AE_test.py`](../scripts/AE_test.py) | 메인 벤치마크 스크립트, 모든 데이터셋에 대해 두 구현 비교 |
| [`install.sh`](../scripts/install.sh) | 빌드 자동화 |
| [`download_dataset.sh`](../scripts/download_dataset.sh) | SuiteSparse 데이터셋 다운로드 |
| [`matrixSampling.py`](../scripts/matrixSampling.py) | 대형 행렬 샘플링 |

---

## Data Flow

### Matrix → Ray/Sphere 변환

```
┌─────────────────────────────────────────────────────────────┐
│                    Matrix A (COO format)                    │
│                                                             │
│   .mtx file:                                                │
│   %%MatrixMarket matrix coordinate real general             │
│   4 4 6                                                     │
│   1 2 3.0    →  Ray: origin=(1, 0, 0), payload=(0,1,3.0)   │
│   2 1 4.0    →  Ray: origin=(0, 0, 0), payload=(1,0,4.0)   │
│   ...                                                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Matrix B (COO format)                    │
│                                                             │
│   1 3 2.0    →  Sphere: center=(0, 2, 0), radius=0.1       │
│   2 1 5.0    →  Sphere: center=(1, 0, 0), radius=0.1       │
│   ...                                                       │
│                                                             │
│   sphereColor[i] = value of i-th non-zero element           │
└─────────────────────────────────────────────────────────────┘
```

### SpMSpM 연산 과정

```
1. Ray Generation
   ┌──────────────────────────────────────────┐
   │ For each non-zero A[i,k]:                │
   │   Launch ray from column k               │
   │   Direction: toward all rows             │
   │   Payload: (row_i, col_k, value_A)       │
   └──────────────────────────────────────────┘
                      │
                      ▼
2. RT Core Traversal (Hardware Accelerated)
   ┌──────────────────────────────────────────┐
   │ BVH traversal on GAS                     │
   │ Find all spheres at column k             │
   │ (These are B[k,*] elements)              │
   └──────────────────────────────────────────┘
                      │
                      ▼
3. Any-Hit Processing
   ┌──────────────────────────────────────────┐
   │ For each intersection (A[i,k], B[k,j]):  │
   │   result = A[i,k].value × B[k,j].value   │
   │   atomicAdd(C[i,j], result)              │
   └──────────────────────────────────────────┘
                      │
                      ▼
4. Result Collection
   ┌──────────────────────────────────────────┐
   │ Dense result buffer → MTX sparse output  │
   └──────────────────────────────────────────┘
```

---

## Key Algorithms

### 1. GAS (Geometry Acceleration Structure) 빌드

```cpp
// optixSpMSpM.cpp:189-260
void buildGAS(optixState& state) {
    // 1. Sphere primitive 설정
    OptixBuildInput sphere_input = {};
    sphere_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    sphere_input.sphereArray.vertexBuffers = &state.devicePoints;  // 중심점
    sphere_input.sphereArray.radiusBuffers = &state.deviceRadius;  // 반경

    // 2. BVH 빌드 옵션
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION
                             | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;

    // 3. GAS 빌드 및 압축
    optixAccelBuild(...);
    optixAccelCompact(...);  // 메모리 최적화
}
```

### 2. Ray-Sphere Intersection 활용

```cuda
// optixSpMSpM.cu:220-282
extern "C" __global__ void __anyhit__ch() {
    // 1. 충돌한 Sphere 정보 가져오기
    unsigned int sphere_idx = optixGetPrimitiveIndex();
    float4 sphere;
    optixGetSphereData(gas, sphere_idx, sbtGASIndex, 0.f, &sphere);

    // 2. Ray payload에서 Matrix A 정보
    float3 payload = getPayload();  // (row_A, col_A, value_A)

    // 3. 곱셈 및 결과 누적
    float result = payload.z * hit_data->sphereColor[sphere_idx];
    uint64_t idx = (uint64_t)payload.x * numCol + (uint64_t)sphere.y;
    atomicAdd(&(hit_data->result[idx]), result);

    // 4. 계속 탐색 (다른 intersection 찾기)
    optixIgnoreIntersection();
}
```

---

## Build & Run

### 빌드

```bash
# Docker 환경 (권장)
cd Dockerfile && ./build_image.sh && ./start_image.sh && ./run.sh

# 수동 빌드
cd /home/RTSpMSpM/scripts && ./install.sh

# 또는 개별 빌드
# OptiX 버전
mkdir -p /home/RTSpMSpM/optixSpMSpM/build
cd /home/RTSpMSpM/optixSpMSpM/build
cmake ../src && make

# cuSPARSE 버전
cd /home/RTSpMSpM/cuSparse/src && make
```

### 실행

```bash
# OptiX SpMSpM
./bin/optixSpMSpM -m1 "matrix1.mtx" -m2 "matrix2.mtx" -o "result.mtx"

# cuSPARSE baseline
./cuSparse -m1 "matrix1.mtx" -m2 "matrix2.mtx" -o "result.mtx"

# 벤치마크 실행
python3 /home/RTSpMSpM/scripts/AE_test.py
```

---

## Code Review Checklist

### 필수 리뷰 항목

#### 1. 핵심 알고리즘 (High Priority)
- [ ] `optixSpMSpM.cu:__anyhit__ch()` - 곱셈 로직 및 atomicAdd 사용 검증
- [ ] `optixSpMSpM.cu:__raygen__rg()` - Ray 생성 로직
- [ ] `optixSpMSpM.cpp:buildGAS()` - GAS 구조 및 최적화 플래그

#### 2. 데이터 변환 (High Priority)
- [ ] `Sphere.cpp:Sphere()` - MTX → Sphere 변환 정확성
- [ ] `optixSpMSpM.cpp:mat1ToGPU()` - MTX → Ray 변환 정확성
- [ ] 0-based vs 1-based 인덱싱 처리 확인

#### 3. 메모리 관리 (Medium Priority)
- [ ] GPU 메모리 할당/해제 누수 확인
- [ ] 결과 버퍼 크기 계산 (`m_result_dim`)
- [ ] 대형 행렬에서의 오버플로우 가능성

#### 4. 성능 고려사항 (Medium Priority)
- [ ] atomicAdd 경합 가능성
- [ ] GAS 압축 효율성
- [ ] Ray 방향 및 tmax 설정

#### 5. 에러 처리 (Low Priority)
- [ ] CUDA/OptiX 에러 체크 매크로 사용
- [ ] 파일 I/O 에러 처리

### 코드 품질 체크

| 항목 | 파일 | 상태 |
|-----|------|------|
| 메모리 누수 | optixSpMSpM.cpp:630-646 | ✅ Cleanup 존재 |
| 에러 처리 | 전체 | ✅ `CUDA_CHECK`, `OPTIX_CHECK` 매크로 |
| 인덱싱 | optixSpMSpM.cu:254-258 | ⚠️ 경계 검사 있음 |
| 동기화 | optixSpMSpM.cpp:609 | ✅ `cudaStreamSynchronize` |

---

## 추가 리소스

- [NVIDIA OptiX Programming Guide](https://raytracing-docs.nvidia.com/)
- [SuiteSparse Matrix Collection](https://sparse.tamu.edu/)
- [cuSPARSE Documentation](https://docs.nvidia.com/cuda/cusparse/)

---

*Last updated: 2025-12-02*
