# ==============================================================================
# [PROJECT] Fluidic Network Grid - MoE Infrastructure Insertion Adapter V2.0
# [FILE] fng_moe_config.py
# [APPLICATION TARGET] Mixtral-8x7B / DeepSeek-V3 MoE Backbone Gating Layer
# ==============================================================================

import os
from typing import List, Final

# ==============================================================================
# 1. HARDWARE ISOLATION & ACCELERATOR TUNING FLAGS
# ==============================================================================
# JAX의 탐욕적 가속기 메모리 사전 할당(Default 90%)을 차단하여 PyTorch KV 캐시 공간을 절연
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# XLA 최적화 컴파일러 내부 융합 그래프 레벨 및 비동기 스트림 인터레이싱 설정
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_gemm=true "              # Triton 고속 행렬 연산 엔진 개방
    "--xla_gpu_graph_level=3 "                        # CUDA Graph 최고 단계 융합 (호스트 오버헤드 소멸)
    "--xla_gpu_enable_highest_priority_async_stream=true "  # PyTorch-JAX 스트림 간 데드락 방지
    "--xla_gpu_memory_limit_slop_bytes=1073741824 "   # OOM 방지를 위한 1GB 가속기 완충 지대 확보
    "--xla_gpu_enable_latency_hiding_scheduler=true"  # 통신 지연을 연산 타임라인 뒤로 은닉
)

# 로컬 예열 및 다중 가속기 토폴오지 강제 에뮬레이션 드라이브 (실배포 환경에선 스케일링 설정에 맞춤)
if "XLA_FORCE_HOST_PLATFORM_DEVICE_COUNT" not in os.environ:
    os.environ["XLA_FLAGS"] += " --xla_force_host_platform_device_count=8"

# ==============================================================================
# 2. PERMANENT SILICON HARDWARE PARAMETERS (COMPILATION ANCHORS)
# ==============================================================================
# 물리 가속기 노드 개수와 1:1 대응하여 NCCL 분기를 소멸시키는 전문가 격자 사양
NUM_EXPERTS: Final[int] = 8

# Llama3 / DeepSeek-V3 백본의 가속기 표준 숨겨진 차원 규격 가이드
FEATURE_DIM: Final[int] = 4096

# ==============================================================================
# 3. DYNAMIC SHAPE BUCKETING CONFIGURATION (INFERENCE ANCHORS)
# ==============================================================================
# 추론 가변 시퀀스 토큰 스트림 유입 시, XLA Re-compilation 렉을 방지하기 위한 정적 컴파일 경계 풀
# 2의 거듭제곱 단위(Bit-shifting Bucket) 축을 기준으로 하여 아키텍처적 연산 효율 극대화
BUCKET_SIZES: Final[List[int]] = [64, 128, 256, 512, 1024, 2048, 4096]

def get_tokens_per_expert(bucket_size: int) -> int:
    """
    각 버킷 크기에 대응하여 전문가 레인당 할당될 정적 가속기 레지스터 슬롯 용량을 산출합니다.
    최소 가동 하한선은 16개 토큰 격자로 제한합니다.
    """
    return max(16, bucket_size // NUM_EXPERTS)

print(f"🔒 [FNG CONFIG] Hardware Parameters Frozen. (Experts: {NUM_EXPERTS}, Feature Dim: {FEATURE_DIM})")
print(f"🔮 [FNG CONFIG] Dynamic Buckets Registered: {BUCKET_SIZES}")
