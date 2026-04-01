import torch
import torch.nn as nn

class Model(nn.Module):
    """
    RWKV Time Decay Exponential with Numerical Stabilization Module.

    Computes time decay with exponential stabilization to prevent
    numerical overflow/underflow in recurrent state updates.

    Key computations:
    1. time_decay_exp = -exp(time_decay)
    2. For each timestep: max_state = max(max_state + time_decay_exp, current_key)
    3. Exponential normalization: e1 = exp(old_max - new_max), e2 = exp(key - new_max)
    """
    def __init__(self):
        super(Model, self).__init__()
        self.attention_hidden_size = 2048

    def forward(
        self,
        time_decay: torch.Tensor,
        key: torch.Tensor,
        time_first: torch.Tensor,
        value: torch.Tensor,
        max_state: torch.Tensor,
        num_state: torch.Tensor,
        den_state: torch.Tensor,
    ):
        """
        RWKV Time Decay Exponential with Numerical Stabilization.

        Args:
            time_decay: Decay rates [attention_hidden_size]
            key: Key tensor [batch_size, seq_len, attention_hidden_size]
            time_first: First time weights [attention_hidden_size]
            value: Value tensor [batch_size, seq_len, attention_hidden_size]
            max_state: Maximum state [batch_size, attention_hidden_size]
            num_state: Numerator state [batch_size, attention_hidden_size]
            den_state: Denominator state [batch_size, attention_hidden_size]

        Returns:
            output: Output tensor [batch_size, seq_len, attention_hidden_size]
            max_state: Updated maximum state
            num_state: Updated numerator state
            den_state: Updated denominator state
        """
        batch_size, seq_len, hidden_size = key.size()

        max_state = max_state.clone().float()
        num_state = num_state.clone().float()
        den_state = den_state.clone().float()

        time_decay_exp = -torch.exp(time_decay.float())

        output = torch.zeros_like(key, dtype=torch.float32)

        for t in range(seq_len):
            current_key = key[:, t].float()
            current_value = value[:, t].float()

            max_for_output = torch.maximum(
                max_state, current_key + time_first
            )

            e1_output = torch.exp(max_state - max_for_output)
            e2_output = torch.exp(current_key + time_first - max_for_output)

            numerator = e1_output * num_state + e2_output * current_value
            denominator = e1_output * den_state + e2_output
            output[:, t] = numerator / denominator

            max_for_state = torch.maximum(
                max_state + time_decay_exp, current_key
            )

            e1_state = torch.exp(max_state + time_decay_exp - max_for_state)
            e2_state = torch.exp(current_key - max_for_state)

            num_state = e1_state * num_state + e2_state * current_value
            den_state = e1_state * den_state + e2_state
            max_state = max_for_state

        return output, max_state, num_state, den_state
