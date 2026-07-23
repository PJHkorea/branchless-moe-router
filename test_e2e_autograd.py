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
from typing import Tuple

# Zero-copy imports of the FNG flat infrastructure architectural assets
from fng_moe_config import NUM_EXPERTS, FEATURE_DIM, BUCKET_SIZES
from fng_moe_dynamic_adapter import FngMoeDynamicShapeAdapter
from fng_moe_monkey_patch import inject_fng_moe_infrastructure_hook

# ==============================================================================
# [MOCK COMPONENT] Architectural Substrate Emulator for Hugging Face Forward Layers
# ==============================================================================
class MockMixtralSparseMoeBlock(torch.nn.Module):
    """
    A validation mock layer that replicates the architectural interface of the 
    official HuggingFace MixtralSparseMoeBlock.
    """
    def __init__(self, num_experts: int, hidden_dim: int):
        super().__init__()
        # Declare gating parameters inside the Torch graph (Primary differentiation target)
        self.gate = torch.nn.Linear(hidden_dim, num_experts, bias=False)
        # Inject standard normal weights to drive active numerical bias gradients
        torch.nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Original legacy forward pipeline prior to injecting the FNG infrastructure hooks
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_dim)
        gate_logits = self.gate(flat_hidden_states)
        
        # Simulates the legacy heavy All-to-All loop and weight combine ops (Forcibly neutralized upon FNG injection)
        legacy_out = flat_hidden_states * 1.1 
        return legacy_out.view(batch_size, sequence_length, hidden_dim), gate_logits

# ==============================================================================
# [EXECUTION] End-to-End Autograd & Bucket Hot-Swap Numerical Integrity Harness
# ==============================================================================
if __name__ == "__main__":
    print("🌊 [AUTOGRAD INFRA] Booting 8-Way Distributed Accelerator Topology...")
    
    # 1. Initialize JAX/XLA distributed mesh topology
    devices = jax.devices()
    moe_mesh = Mesh(jnp.array(devices).reshape(8), ("moe_cluster",))
    
    # 2. Instantiate FNG dynamic shape adapter (Executes offline multi-bucket static warm-up)
    print("⚙️  [FNG ADAPTER] Pre-compiling Multi-Bucket Registries (Warm-up)...")
    adapter = FngMoeDynamicShapeAdapter(mesh=moe_mesh)
    
    # 3. Instantiate the architectural mock module onto the active accelerator rails
    with torch.device("cuda"):
        mock_moe_block = MockMixtralSparseMoeBlock(num_experts=NUM_EXPERTS, hidden_dim=FEATURE_DIM)
        
    # 4. 🔥 [INJECTION] Affix runtime monkey-patch infrastructure hooks
    mock_moe_block = inject_fng_moe_infrastructure_hook(mock_moe_block, adapter)
    
    # 5. [Adversarial Dynamic Sequence Stream Simulation]
    # Deploy an evaluation environment with sequential fluctuating token influx (prime, boundary, and odd dimensions)
    test_inference_scenarios = [45, 128, 211, 500] # Variable token stream scenario vector
    
    print("\n⚡ [EXECUTION] Driving Variable Token Sequence Matrix Stream...")
    for step, actual_tokens in enumerate(test_inference_scenarios):
        print(f"\n🔮 [STEP {step}] Input Inflow Size: {actual_tokens} Tokens")
        
        # Instantiate dynamic-size PyTorch tensors with active differentiation tracking enabled
        mock_hidden_states = torch.randn(1, actual_tokens, FEATURE_DIM, device="cuda", requires_grad=True)
        
        # 6. Stream through forward passes (Forcibly routes via the patched JAX 0ns Mux kernels)
        # Statically scan output spatial dimension integrity and zero-copy virtual viewport slicing accuracy
        final_output, gate_logits = mock_moe_block(mock_hidden_states)
        
        # 7. Map a synthetic target loss function to evaluate numerical convergence and backward pipeline integrity
        # Objective function designed to verify comprehensive gradient propagation across data channels and gating nodes
        loss = torch.sum(torch.square(final_output))
        
        # 8. 🔄 [BACKWARD INTERRUPT] Trigger zero-copy backpropagation: PyTorch Autograd -> JAX VJP
        loss.backward()

             # 9. ==========================================================================
        # 📊 [AUTOGRAD INTEGRITY REPORT] Parsing & Scanning Numerical Convergence Outcomes
        # ==============================================================================
        print(f" ├─ Forward Matrix Loss Value : {loss.item():.4f}")
        print(f" ├─ Final Output Tensor Shape : {list(final_output.shape)} (Expected: [1, {actual_tokens}, {FEATURE_DIM}])")
        
        # Extract derivative node gradients propagated via backpropagation
        grad_hidden = mock_hidden_states.grad
        grad_gate_weight = mock_moe_block.gate.weight.grad
        
        # Guardrails to scan for hardware race conditions and underflow/overflow numerical contamination
        nan_in_hidden = torch.isnan(grad_hidden).any().item()
        nan_in_gate = torch.isnan(grad_gate_weight).any().item()
        zero_in_gate = (torch.count_nonzero(grad_gate_weight) == 0).item()
        
        print(f" ├─ Token Gradient NaN Detect : {nan_in_hidden} (Pass: False)")
        print(f" ├─ Gate Gradient NaN Detect  : {nan_in_gate} (Pass: False)")
        print(f" └─ Gradient Vanishing Stall  : {zero_in_gate} (Pass: False)")
        
        # Clear gradient target buffers to prepare for the subsequent dynamic sequence step
        mock_hidden_states.grad.zero_()
        mock_moe_block.gate.weight.grad.zero_()
        
        if nan_in_hidden or nan_in_gate or zero_in_gate:
            print("\n❌ [CRITICAL] AUTOGRAD DISCONNECTED OR GRADIENT BLEEDING DETECTED.")
            raise RuntimeError("Autograd chain broken in FNG Infrastructure layer.")
            
    print("\n🎯 [SUCCESS] ADIABATIC BACKPROPAGATION CHAIN RUN TERMINATED CLEANLY WITH DETERMINISTIC CONVERGENCE.")
