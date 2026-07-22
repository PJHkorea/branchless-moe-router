# ==============================================================================
# [PROJECT] Fluidic Network Grid - MoE Infrastructure Insertion Adapter V2.0
# [FILE] fng_moe_dynamic_adapter.py
# [APPLICATION TARGET] Mixtral-8x7B / DeepSeek-V3 MoE Backbone Gating Layer
# ==============================================================================

import jax
import jax.numpy as jnp
import torch
from typing import Dict, Any, Tuple

# 전역 설정 및 핵심 팩토리 커널 임포트
from fng_moe_config import NUM_EXPERTS, FEATURE_DIM, BUCKET_SIZES, get_tokens_per_expert
from fng_moe_core_kernel import create_fng_moe_autograd_pipeline
from fng_moe_autograd_bridge import FngMoeAutogradBridge

class FngMoeDynamicShapeAdapter:
    def __init__(self, mesh: jax.sharding.Mesh):
        """
        [DYNAMIC INFERENCE INFRA] 
        가변 시퀀스 추론 가속화를 위해 동적 버킷 격리 레이어를 탑재한 MoE 어댑터 코어.
        """
        self.mesh = mesh
        self.bucket_sizes = sorted(BUCKET_SIZES)
        self.max_global_tokens = self.bucket_sizes[-1]
        
        # 각 버킷 크기별로 최적화된 팩토리 라우터 커널을 독립적으로 JIT 사전 동결 컴파일
        # 런타임에는 조건문 분기 없이 딕셔너리 매핑(0ns 주소 호출)으로 실행 커널을 핫스왑합니다.
        self.router_bucket_registry = {}
        self._precompile_all_buckets()
        print(f"🔒 [FNG ADAPTER] Dynamic-Shape Buckets Registered and Frozen: {self.bucket_sizes}")

    def _precompile_all_buckets(self):
        """
        오프라인 환경에서 각 버킷 규격에 맞는 XLA 대수적 멀티플렉서 그래프를 영구 동결합니다.
        """
        for bucket_size in self.bucket_sizes:
            # 버킷 크기에 비례하여 전문가당 수용 가능한 정적 레지스터 크기를 가변 조정
            tokens_per_expert = get_tokens_per_expert(bucket_size)
            
            # 독립된 컴파일 실행 객체 생성 및 레지스트리 영구 등록
            raw_pipeline = create_fng_moe_autograd_pipeline(tokens_per_expert)
            self.router_bucket_registry[bucket_size] = raw_pipeline

    def _find_optimal_bucket(self, actual_tokens_count: int) -> int:
        """
        [1클럭 바이너리 서치] 입력 토큰 스트림을 수용할 최적의 정적 컴파일 버킷 경계를 산출합니다.
        """
        for bucket in self.bucket_sizes:
            if actual_tokens_count <= bucket:
                return bucket
        raise ValueError(f"🚨 Input tokens ({actual_tokens_count}) exceeds maximum infrastructure bucket size ({self.max_global_tokens})")

    def inject_dynamic_inference_pass(
        self, 
        hidden_states: torch.Tensor, # [Actual_Tokens, Feature_Dim] (가변 추론 시퀀스 입력)
        gate_logits: torch.Tensor    # [Actual_Tokens, Num_Experts]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        [DYNAMIC INFERENCE ENTRYPOINT]
        KV 캐시 유실을 차단하고 가속기 렉 없이 가변 시퀀스를 소화하는 고속 인퍼런스 패스.
        """
        actual_tokens = hidden_states.shape[0]
        
        # 1. 런타임에 최적의 정적 컴파일 버킷 축 핫스왑 선택 (Re-compilation 전면 차단)
        target_bucket_size = self._find_optimal_bucket(actual_tokens)
        pad_size = target_bucket_size - actual_tokens
        
        # 2. PyTorch 하드웨어 메모리 단에서 초고속 정적 패딩 가동
        if pad_size > 0:
            # 빈 공간은 0.0, 게이팅 로짓은 마스킹을 위해 극단적인 음수(-1e9)로 물리 패딩
            # XLA jnp.argmax가 패딩 영역을 안전하게 더미 주소선으로 격리하도록 유도
            hidden_states_padded = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad_size), value=0.0)
            gate_logits_padded = torch.nn.functional.pad(gate_logits, (0, 0, 0, pad_size), value=-1e9)
        else:
            hidden_states_padded = hidden_states
            gate_logits_padded = gate_logits

        # 3. Autograd 제로카피 미분 체인 및 분산 메시 파이프라인 관류
        # 패딩 처리된 정적 버킷 크기의 실행 커널을 맵에서 0ns 핫스왑 드로우
        target_pipeline = self.router_bucket_registry[target_bucket_size]
        
        torch_combined_padded = FngMoeAutogradBridge.apply(
            self.mesh,
            target_pipeline,
            hidden_states_padded,
            gate_logits_padded
        )
        
        # 4. [0-copy 슬라이싱 복원] 패딩된 더미 영역을 도살하고 실제 원본 토큰 시퀀스만 칼같이 슬라이싱
        # 이 연산은 메모리 카피를 발생시키지 않고 오리지널 가상 뷰 포인터만 재정렬합니다.
        torch_final_out = torch_combined_padded[:actual_tokens, :]
        
        # 하드웨어 비동기 스트림 메모리 파멸 방지용 수명 주기 가드 배치
        torch_final_out._source_tensors = (hidden_states, gate_logits, torch_combined_padded)
        
        return torch_final_out, {"bucket_used": target_bucket_size, "padded_tokens": pad_size}
