# RTSpMSpM Architecture Deep Dive

이 문서는 RTSpMSpM의 아키텍처를 더 깊이 이해하기 위한 상세 가이드입니다.

---

## 1. OptiX Pipeline 상세

### Pipeline 구성 요소

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          OptiX Pipeline                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐                                                       │
│  │   Module     │  optixSpMSpM.cu 컴파일 결과                           │
│  │  (PTX/OPTIX) │  - __raygen__rg, __anyhit__ch, __miss__ms 포함        │
│  └──────────────┘                                                       │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │                    Program Groups                             │       │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │       │
│  │  │  RayGen    │  │   Miss     │  │      HitGroup          │  │       │
│  │  │  Group     │  │   Group    │  │  (AnyHit + BuiltinIS)  │  │       │
│  │  └────────────┘  └────────────┘  └────────────────────────┘  │       │
│  └──────────────────────────────────────────────────────────────┘       │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │              Shader Binding Table (SBT)                       │       │
│  │  ┌─────────────────────────────────────────────────────────┐ │       │
│  │  │ RayGen Record: RayData (originVec, size)                │ │       │
│  │  ├─────────────────────────────────────────────────────────┤ │       │
│  │  │ Miss Record: MissData (unused)                          │ │       │
│  │  ├─────────────────────────────────────────────────────────┤ │       │
│  │  │ HitGroup Record: SphereData (sphereColor, result, dims) │ │       │
│  │  └─────────────────────────────────────────────────────────┘ │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pipeline 옵션 설명

```cpp
// optixSpMSpM.cpp:273-279
state.pipeline_compile_options = {
    .usesMotionBlur = false,
    .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
    .numPayloadValues = 3,        // float3 payload (row, col, value)
    .numAttributeValues = 1,       // sphere 속성
    .pipelineLaunchParamsVariableName = "params",
    .usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE  // Sphere primitive
};
```

---

## 2. GAS (Geometry Acceleration Structure)

### BVH 트리 구조

```
                    ┌─────────────┐
                    │  Root Node  │
                    │  (AABB)     │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
        ┌─────▼─────┐             ┌─────▼─────┐
        │ Internal  │             │ Internal  │
        │   Node    │             │   Node    │
        └─────┬─────┘             └─────┬─────┘
              │                         │
     ┌────────┴────────┐       ┌────────┴────────┐
     │                 │       │                 │
┌────▼────┐      ┌────▼────┐ ┌────▼────┐    ┌────▼────┐
│ Sphere  │      │ Sphere  │ │ Sphere  │    │ Sphere  │
│ B[k1,j1]│      │ B[k1,j2]│ │ B[k2,j1]│    │ B[k2,j2]│
└─────────┘      └─────────┘ └─────────┘    └─────────┘
```

### Sphere Primitive 구성

```cpp
// Sphere 생성 (Sphere.cpp:70-77)
// Matrix B의 각 non-zero 원소 (row, col, value)를 변환:
float3 point = make_float3(row - 1, col - 1, 0.f);  // 0-based 변환
float radius = 0.1f;  // 고정 반경
```

**Sphere 배치 전략:**
- X축: Matrix B의 row index (= k, 공통 차원)
- Y축: Matrix B의 column index (= j, 결과 행렬의 column)
- Z축: 0 (2D 문제이므로 사용 안 함)
- 반경: 0.1 (작은 값으로 정확한 intersection)

---

## 3. Ray 구성 및 Tracing

### Ray 생성 로직

```cuda
// optixSpMSpM.cu:113-197
__raygen__rg() {
    // Matrix A의 각 non-zero 원소에 대해 하나의 Ray 생성
    int ray_idx = optixGetLaunchIndex().x;

    // A[i,k]의 column index k가 ray의 origin이 됨
    float origin_k = ray_data->originVec[ray_idx].y;  // column of A

    // Ray 구성
    float3 origin = make_float3(origin_k, 0.0f, 0.0f);
    float3 direction = make_float3(origin_k, 1e16f, 0.0f);  // Y방향으로 발사

    // Payload: A[i,k] 정보
    float3 payload = ray_data->originVec[ray_idx];  // (row, col, value)

    trace(params.handle, origin, direction, 0.0f, 1e16f, &payload);
}
```

### Ray-Sphere Intersection 의미

```
  Y (Column of B / Result)
  ↑
  │     ● B[k,j1]      ● B[k,j2]      ● B[k,j3]
  │        │              │              │
  │        │              │              │
  │────────┼──────────────┼──────────────┼────────→ Ray from A[i,k]
  │        │              │              │
  │        ↓              ↓              ↓
  │    Intersection   Intersection   Intersection
  │
  └──────────────────────────────────────────────→ X (Row of B / k)

Ray: origin=(k, 0, 0), direction=(k, 1e16, 0)
  → Y 방향으로 쏘아서 같은 row(k)의 모든 B 원소와 교차
```

---

## 4. Memory Layout

### Host → Device 데이터 전송

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              HOST MEMORY                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Matrix A (MTX file)              Matrix B (MTX file)                   │
│  ┌─────────────────────┐          ┌─────────────────────┐               │
│  │ (row, col, value)   │          │ (row, col, value)   │               │
│  │ (1, 2, 3.0)         │          │ (2, 1, 5.0)         │               │
│  │ (2, 1, 4.0)         │          │ (1, 3, 2.0)         │               │
│  │ ...                 │          │ ...                 │               │
│  └─────────────────────┘          └─────────────────────┘               │
│           │                                │                             │
│           │ mat1ToGPU()                    │ storeSphereData()          │
│           ▼                                ▼                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                            cudaMemcpy
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                             DEVICE MEMORY                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  d_matrix (float3*)               devicePoints (float3*)                │
│  ┌─────────────────────┐          ┌─────────────────────┐               │
│  │ (0,1,3.0) → Ray 0   │          │ (1,0,0) → Sphere 0  │               │
│  │ (1,0,4.0) → Ray 1   │          │ (0,2,0) → Sphere 1  │               │
│  └─────────────────────┘          └─────────────────────┘               │
│                                                                          │
│  spherePoints (float*)            d_result (float*)                     │
│  ┌─────────────────────┐          ┌─────────────────────────────┐       │
│  │ 5.0 (value of B[0]) │          │ 0.0 0.0 0.0 ... │ rows×cols │       │
│  │ 2.0 (value of B[1]) │          │ 0.0 0.0 0.0 ... │ (dense)   │       │
│  └─────────────────────┘          └─────────────────────────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Result Buffer 구조

```cpp
// Dense format으로 저장 (optixSpMSpM.cpp:164-171)
state.d_result_buf_size = rows × cols × sizeof(float);

// Any-hit에서 결과 누적 (optixSpMSpM.cu:254-258)
uint64_t idx = rowIndex * numCol + colIndex;  // 1D 인덱스 계산
atomicAdd(&(result[idx]), value);              // 원자적 덧셈
```

---

## 5. cuSPARSE Baseline 비교

### 알고리즘 차이

| 항목 | RTSpMSpM (OptiX) | cuSPARSE |
|-----|------------------|----------|
| 자료구조 | Sphere BVH | CSR |
| 연산 방식 | Ray-Sphere Intersection | Gustavson's algorithm |
| 메모리 접근 | Random (BVH traverse) | Structured (CSR) |
| 결과 포맷 | Dense (then sparse) | Sparse (직접) |
| 하드웨어 | RT Cores | CUDA Cores |

### 성능 특성

```
RTSpMSpM 장점:
- RT Core의 하드웨어 가속 BVH traversal
- 매우 희소한 행렬에서 효율적
- Intersection 검출이 O(log N)

cuSPARSE 장점:
- 범용 GPU에서 동작
- 메모리 효율적 (sparse output)
- 성숙한 최적화
```

---

## 6. 성능 최적화 포인트

### 현재 구현의 특징

```cpp
// GAS Build Flags (optixSpMSpM.cpp:195)
accel_options.buildFlags =
    OPTIX_BUILD_FLAG_ALLOW_COMPACTION |       // 메모리 압축
    OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;  // Sphere 데이터 접근

// Pipeline 설정 (optixSpMSpM.cpp:351)
max_trace_depth = 1;  // 단일 레벨 tracing
```

### 잠재적 최적화 영역

1. **결과 버퍼**: Dense → Sparse 변환 시 오버헤드
2. **atomicAdd 경합**: 동일 C[i,j]에 대한 경쟁
3. **GAS 재사용**: 동일 Matrix B에 대해 GAS 캐싱 가능
4. **배치 처리**: 여러 행렬 연산 일괄 처리

---

## 7. 확장 가능성

### 다른 연산으로의 확장

| 연산 | 매핑 전략 |
|-----|---------|
| SpMV (Sparse × Dense Vector) | Ray = sparse row, Line = dense vector |
| SpMM (Sparse × Dense Matrix) | 여러 dense vector 동시 처리 |
| Masked SpMSpM | 마스크 조건으로 intersection 필터링 |

### 다른 primitive 활용

- **Triangle**: 더 큰 영역 커버 가능
- **Curve**: 연속적인 패턴에 활용
- **Custom AABB**: 임의 형태의 intersection 영역

---

*이 문서는 코드 리뷰 시 아키텍처 이해를 돕기 위해 작성되었습니다.*
