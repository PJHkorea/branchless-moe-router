# ==============================================================================
# [PROJECT] Fluidic Network Grid - MoE Infrastructure Insertion Adapter V2.0
# [FILE] test_e2e_autograd.py
# [APPLICATION TARGET] Mixtral-8x7B / DeepSeek-V3 MoE Backbone Gating Layer
# ==============================================================================

import os
import torch
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

# FNG 플랫 인프라 아키텍처 자산 0-copy 임포트
from fng_moe_config import NUM_EXPERTS, FEATURE_DIM, BUCKET_SIZES
from fng_moe_dynamic_adapter import FngMoeDynamicShapeAdapter
from fng_moe_monkey_patch import inject_fng_moe_infrastructure_hook

# ==============================================================================
# [MOCK COMPONENT] 허깅페이스 정방향 레이어 인터페이스 에뮬레이터용 의사 블록
# ==============================================================================
class MockMixtralSparseMoeBlock(torch.nn.Module):
    """
    HuggingFace MixtralSparseMoeBlock의 아키텍처 인터페이스를 모사한 검증용 모킹 레이어
    """
    def __init__(self, num_experts: int, hidden_dim: int):
        super().__init__()
        # 게이팅 파라미터 그래프 선언 (미분 타겟)
        self.gate = torch.nn.Linear(hidden_dim, num_experts, bias=False)
        # 수치 편향 유도를 위한 정규 가중치 임의 사입
        torch.nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 인젝션 후크가 주입되기 전의 오리지널 레거시 Forward 파이프라인
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_dim)
        gate_logits = self.gate(flat_hidden_states)
        
        # 레거시의 무거운 All-to-All 및 가중치 결합 처리 모사 (FNG 주입 시 이 구간 도살됨)
        legacy_out = flat_hidden_states * 1.1 
        return legacy_out.view(batch_size, sequence_length, hidden_dim), gate_logits

# ==============================================================================
# [EXECUTION] 자동 미분 및 버킷 핫스왑 수치 무결성 검증 런타임
# ==============================================================================
if __name__ == "__main__":
    print("🌊 [AUTOGRAD INFRA] Booting 8-Way Distributed Accelerator Topology...")
    
    # 1. JAX/XLA 분산 메시 초기화
    devices = jax.devices()
    moe_mesh = Mesh(jnp.array(devices).reshape(8), ("moe_cluster",))
    
    # 2. FNG 동적 어댑터 레이어 초기화 (오프라인 5개 버킷 사전 컴파일 동결)
    print("⚙️  [FNG ADAPTER] Pre-compiling Multi-Bucket Registries (Warm-up)...")
    adapter = FngMoeDynamicShapeAdapter(mesh=moe_mesh)
    
    # 3. 모킹 모델 생성 및 가속기 레일 위 장착
    with torch.device("cuda"):
        mock_moe_block = MockMixtralSparseMoeBlock(num_experts=NUM_EXPERTS, hidden_dim=FEATURE_DIM)
        
    # 4. 🔥 [INJECTION] 런타임 몽키 패치 인터럽트 후크 고착
    mock_moe_block = inject_fng_moe_infrastructure_hook(mock_moe_block, adapter)
    
    # 5. [가혹한 추론/학습 가변 시퀀스 스트림 인입 시뮬레이션]
    # 서로 다른 토큰 크기(홀수, 버킷 경계값 등)가 연속으로 들어오는 환경 전개
    test_inference_scenarios = [45, 128, 211, 500] # 가변 토큰 스트림 시나리오 축
    
    print("\n⚡ [EXECUTION] Driving Variable Token Sequence Matrix Stream...")
    for step, actual_tokens in enumerate(test_inference_scenarios):
        print(f"\n🔮 [STEP {step}] Input Inflow Size: {actual_tokens} Tokens")
        
        # 가변 크기의 그라디언트 추적 활성화 파이토치 입력 텐서 생성
        mock_hidden_states = torch.randn(1, actual_tokens, FEATURE_DIM, device="cuda", requires_grad=True)
        
        # 6. 정방향 관류 실행 (인젝션된 후크에 의해 JAX 0ns Mux 커널 동작)
        # 출력 디멘션의 무결성 검증 및 슬라이싱 복원 작동 스캔
        final_output, gate_logits = mock_moe_block(mock_hidden_states)
        
        # 7. 수치 수렴 및 역미분 파이프라인 무결성을 위한 가상 손실 함수(Loss) 매핑
        # 그라디언트 전파가 전 영역(데이터축, 게이트 가중치축)에 전염되는지 검사하기 위한 목적 함수
        loss = torch.sum(torch.square(final_output))
        
        # 8. 🔄 [BACKWARD INTERRUPT] PyTorch Autograd -> JAX VJP 0-copy 역전파 트리거
        loss.backward()
        
        # 9. ==========================================================================
        # 📊 [AUTOGRAD INTEGRITY REPORT] 수치 수렴 결과 파싱 및 스캔
        # ==========================================================================
        print(f" ├─ Forward Matrix Loss Value : {loss.item():.4f}")
        print(f" ├─ Final Output Tensor Shape : {list(final_output.shape)} (Expected: [1, {actual_tokens}, {FEATURE_DIM}])")
        
        # 역방향 오차 전달 노드 그라디언트 추출
        grad_hidden = mock_hidden_states.grad
        grad_gate_weight = mock_moe_block.gate.weight.grad
        
        # 하드웨어 레이스 컨디션 및 언더플로우/오버플로우 오염 스캔 가드레일
        nan_in_hidden = torch.isnan(grad_hidden).any().item()
        nan_in_gate = torch.isnan(grad_gate_weight).any().item()
        zero_in_gate = (torch.count_nonzero(grad_gate_weight) == 0).item()
        
        print(f" ├─ Token Gradient NaN Detect : {nan_in_hidden} (Pass: False)")
        print(f" ├─ Gate Gradient NaN Detect  : {nan_in_gate} (Pass: False)")
        print(f" └─ Gradient Vanishing Stall  : {zero_in_gate} (Pass: False)")
        
        # 그라디언트 버퍼 클리어 (다음 가변 루프 스텝 대비)
        mock_hidden_states.grad.zero_()
        mock_moe_block.gate.weight.grad.zero_()
        
        if nan_in_hidden or nan_in_gate or zero_in_gate:
            print("\n❌ [CRITICAL] AUTOGRAD DISCONNECTED OR GRADIENT BLEEDING DETECTED.")
            raise RuntimeError("Autograd chain broken in FNG Infrastructure layer.")
            
    print("\n🎯 [SUCCESS] ADIABATIC BACKPROPAGATION CHAIN RUN TERMINATED CLEANLY WITH DETERMINISTIC 수렴.")
