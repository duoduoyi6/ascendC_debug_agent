import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Tanh-gated Residual Addition Backward Module.

    Backward pass for tanh-gated residual addition.
    Forward was: output = residual + tanh(gate) * hidden_states * mask

    Gradients:
    - grad_residual = grad_output (identity)
    - grad_hidden_states = grad_output * tanh(gate) * mask
    - grad_gate = sum(grad_output * hidden_states * mask * sech^2(gate))
              = sum(grad_output * hidden_states * mask * (1 - tanh^2(gate)))
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, gate: torch.Tensor, hidden_states: torch.Tensor, mask: torch.Tensor):
        """
        Backward pass for tanh-gated residual addition.

        Args:
            grad_output: Gradient of loss w.r.t. output
            gate: Gate tensor from forward pass
            hidden_states: Hidden states from forward pass
            mask: Mask tensor from forward pass

        Returns:
            grad_residual: Gradient w.r.t. residual (same as grad_output)
            grad_hidden_states: Gradient w.r.t. hidden_states
            grad_gate: Gradient w.r.t. gate (scalar)
        """
        gate_float = gate.to(torch.float32)
        gate_value = torch.tanh(gate_float)
        grad_residual = grad_output.clone()
        grad_hidden_states = grad_output * gate_value * mask
        sech_squared = 1.0 - gate_value * gate_value
        masked_hidden_states = hidden_states * mask
        grad_gate = torch.sum(grad_output.to(torch.float32) * masked_hidden_states.to(torch.float32)) * sech_squared
        return grad_residual.to(torch.bfloat16), grad_hidden_states.to(torch.bfloat16), grad_gate.to(torch.bfloat16)
