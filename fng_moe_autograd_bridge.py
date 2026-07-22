# ==============================================================================
# [PROJECT] Fluidic Network Grid - MoE Infrastructure Insertion Adapter V2.0
# [FILE] fng_moe_autograd_bridge.py
# [APPLICATION TARGET] Mixtral-8x7B / DeepSeek-V3 MoE Backbone Gating Layer
# ==============================================================================

import jax
import torch
from typing import Tuple, Any

class FngMoeAutogradBridge(torch.autograd.Function):
    """
    [HARDWARE ADAPTER BOUNDARY] 
    PyTorch Autograd 타임라인과 JAX 분산 메시 가속기 기계어 간의 
    미분 사슬을 복사 비용 0바이트로 1:1 결합하는 프로덕션 엔지니어링 브릿지.
    """
    
    @staticmethod
    def forward(
        ctx: Any, 
        mesh: jax.sharding.Mesh,
        e2e_pipeline: Any,          # create_fng_moe_autograd_pipeline() 팩토리 인스턴스
        hidden_states: torch.Tensor, # [Global_Tokens, Feature_Dim]
        gate_logits: torch.Tensor    # [Global_Tokens, Num_Experts]
    ) -> torch.Tensor:
        """
        [FORWARD INTERRUPT] 
        파이토치 텐서를 JAX 레일 위로 밀어 넣어 가상화 라우팅 후 결과만 리턴하며,
        역전파를 위한 VJP(Vector-Jacobian Product) 추적용 하드웨어 펜스를 컨텍스트에 박제합니다.
        """
        # 1. 하드웨어 메모리 배치 연속성(Contiguous) 보장
        if not hidden_states.is_contiguous(): 
            hidden_states = hidden_states.contiguous()
        if not gate_logits.is_contiguous(): 
            gate_logits = gate_logits.contiguous()
        
        # 2. NVIDIA DLPack 메커니즘을 통한 가속기 포인터 수준 가로채기 (0-copy)
        jax_tokens = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(hidden_states))
        jax_logits = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(gate_logits))
        
        # 3. XLA VJP 미분 사슬을 추적 모드로 빌드하기 위해 jax.vjp 엔진 가동
        # _e2e_vjp_fn은 역전파 시 가속기 내부 레지스터 그라디언트를 역산할 가상 기계어 주소선입니다.
        with mesh:
            def _pure_jax_forward(t, l):
                return e2e_pipeline(mesh, t, l)
                
            jax_outputs, _e2e_vjp_fn = jax.vjp(_pure_jax_forward, jax_tokens, jax_logits)
            
        # 4. 결과물을 파이토치 전문가 MLP 백엔드로 복사 없이 바이패스 반환
        torch_outputs = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jax_outputs))
        
        # 5. [Autograd Guard] 파이토치 가비지 컬렉터의 메모리 오염 차단 및 JAX VJP 함수 이식
        ctx.mesh = mesh
        ctx._e2e_vjp_fn = _e2e_vjp_fn
        ctx.save_for_backward(hidden_states, gate_logits)
        
        # 가속기 연산 도중 PyTorch가 입력 메모리를 덮어쓸 위험을 방어하는 수명 주기 확장 가드
        torch_outputs._source_tensors = (hidden_states, gate_logits) 
        
        return torch_outputs

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[None, None, torch.Tensor, torch.Tensor]:
        """
        [BACKWARD INTERRUPT]
        상위 레이어에서 유입된 파이토치 오차 그라디언트를 가로채, 
        동결 컴파일된 JAX VJP 기전으로 관류시켜 토큰과 게이팅 로짓의 그라디언트를 역산출합니다.
        """
        mesh = ctx.mesh
        _e2e_vjp_fn = ctx._e2e_vjp_fn
        
        # 1. 상위 오차 그라디언트 0-copy 변환 및 연속성 가드 설치
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        jax_grad_output = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(grad_output))
        
        # 2. 박제해 두었던 XLA VJP 타임라인 호출하여 양방향 부분미분 그라디언트 역배분
        # 하드웨어 레벨에서 수치적 수렴 및 누수 제로(NaN 배제) 연산 즉시 가동
        with mesh:
            jax_grad_tokens, jax_grad_logits = _e2e_vjp_fn(jax_grad_output)
            
        # 3. 계산 완료된 하드웨어 그라디언트를 파이토치 Autograd 그래프 노드로 0바이트 전송
        torch_grad_tokens = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jax_grad_tokens))
        torch_grad_logits = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jax_grad_logits))
        
        # 4. 입력 인자 축의 순서에 정확히 맞추어 그라디언트 튜플 반환 
        # (mesh, e2e_pipeline 인자는 미분 대상이 아니므로 명시적 None 마킹 필수)
        return None, None, torch_grad_tokens, torch_grad_logits
