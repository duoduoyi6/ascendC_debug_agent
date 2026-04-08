import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Rotary Position Embedding multiplication.
    Implements the same computation as torch_npu.npu_rotary_mul using pure PyTorch.

    For 'half' mode:
        output = input * r1 + rotate_half(input) * r2
        where rotate_half swaps the two halves of the last dimension and negates the first half.

    For 'interleave' mode:
        output = input * r1 + rotate_interleave(input) * r2
        where rotate_interleave interleaves pairs: [-x1, x0, -x3, x2, ...].
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, rotary_mode: str = 'half') -> torch.Tensor:
        """
        Applies rotary position embedding multiplication to the input tensor.

        Args:
            input (torch.Tensor): Input tensor, must be 4D. Supports float16, bfloat16, float32.
            r1 (torch.Tensor): Cosine rotation coefficient, must be 4D. Supports float16, bfloat16, float32.
            r2 (torch.Tensor): Sine rotation coefficient, must be 4D. Supports float16, bfloat16, float32.
            rotary_mode (str, optional): Computation mode, supports 'half' and 'interleave'. Default: 'half'.

        Returns:
            torch.Tensor: Output tensor with rotary position embedding applied.
        """
        if rotary_mode == 'half':
            half_d = input.shape[-1] // 2
            x1 = input[..., :half_d]
            x2 = input[..., half_d:]
            rotated = torch.cat((-x2, x1), dim=-1)
        elif rotary_mode == 'interleave':
            x1 = input[..., 0::2]
            x2 = input[..., 1::2]
            rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
        else:
            raise ValueError(f"Unsupported rotary_mode: {rotary_mode}")
        return input * r1 + rotated * r2

INPUT_CASES = [{'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 1, 512, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 512, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 512, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 1, 1024, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 1024, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 1024, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 1, 2048, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 2048, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 2048, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 1, 128, 128], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 128, 128], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 128, 128], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 1, 256, 128], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 256, 128], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 256, 128], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 1, 512, 128], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 512, 128], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 512, 128], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 1, 1024, 128], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 1024, 128], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 1024, 128], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'interleave'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'interleave'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 1, 512, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 512, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 512, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'interleave'}]},
 {'inputs': [{'dtype': 'bfloat16', 'name': 'input', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r1', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r2', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'bfloat16', 'name': 'input', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r1', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r2', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'bfloat16', 'name': 'input', 'required': True, 'shape': [1, 1, 512, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r1', 'required': True, 'shape': [1, 1, 512, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r2', 'required': True, 'shape': [1, 1, 512, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'bfloat16', 'name': 'input', 'required': True, 'shape': [1, 1, 1024, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r1', 'required': True, 'shape': [1, 1, 1024, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r2', 'required': True, 'shape': [1, 1, 1024, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'bfloat16', 'name': 'input', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r1', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r2', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'interleave'}]},
 {'inputs': [{'dtype': 'bfloat16', 'name': 'input', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r1', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r2', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'interleave'}]},
 {'inputs': [{'dtype': 'float32', 'name': 'input', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float32', 'name': 'r1', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float32', 'name': 'r2', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float32', 'name': 'input', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'float32', 'name': 'r1', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'float32', 'name': 'r2', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float32', 'name': 'input', 'required': True, 'shape': [1, 1, 512, 64], 'type': 'tensor'},
             {'dtype': 'float32', 'name': 'r1', 'required': True, 'shape': [1, 1, 512, 64], 'type': 'tensor'},
             {'dtype': 'float32', 'name': 'r2', 'required': True, 'shape': [1, 1, 512, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float32', 'name': 'input', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float32', 'name': 'r1', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float32', 'name': 'r2', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'interleave'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [2, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [2, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [2, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [4, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [4, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [4, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [8, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [8, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [8, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [16, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [16, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [16, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 8, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 16, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 32, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [2, 8, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [2, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [2, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [4, 8, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [4, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [4, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 8, 256, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 8, 512, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 512, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 512, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 8, 1024, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 1024, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 1024, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 8, 2048, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 2048, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 2048, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 8, 128, 128], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 128, 128], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 128, 128], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 8, 256, 128], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 256, 128], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 256, 128], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 8, 512, 128], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 512, 128], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 512, 128], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'bfloat16', 'name': 'input', 'required': True, 'shape': [1, 8, 128, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r1', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r2', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'bfloat16', 'name': 'input', 'required': True, 'shape': [1, 8, 256, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r1', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r2', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'bfloat16', 'name': 'input', 'required': True, 'shape': [1, 8, 512, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r1', 'required': True, 'shape': [1, 1, 512, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r2', 'required': True, 'shape': [1, 1, 512, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'bfloat16', 'name': 'input', 'required': True, 'shape': [1, 8, 1024, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r1', 'required': True, 'shape': [1, 1, 1024, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r2', 'required': True, 'shape': [1, 1, 1024, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 8, 4096, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 4096, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 4096, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 8, 8192, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 8192, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 8192, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 8, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 128, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'interleave'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 8, 256, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 256, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'interleave'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 8, 512, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 512, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 512, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'interleave'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 8, 16384, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 16384, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 16384, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'float16', 'name': 'input', 'required': True, 'shape': [1, 8, 32768, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r1', 'required': True, 'shape': [1, 1, 32768, 64], 'type': 'tensor'},
             {'dtype': 'float16', 'name': 'r2', 'required': True, 'shape': [1, 1, 32768, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'half'}]},
 {'inputs': [{'dtype': 'bfloat16', 'name': 'input', 'required': True, 'shape': [1, 8, 32768, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r1', 'required': True, 'shape': [1, 1, 32768, 64], 'type': 'tensor'},
             {'dtype': 'bfloat16', 'name': 'r2', 'required': True, 'shape': [1, 1, 32768, 64], 'type': 'tensor'},
             {'dtype': 'str', 'name': 'rotary_mode', 'required': False, 'type': 'attr', 'value': 'interleave'}]}]

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _make_boxes(shape, dtype):
    leading_shape = tuple(shape[:-1])
    mins = torch.rand(*leading_shape, 2, dtype=torch.float32)
    sizes = torch.rand(*leading_shape, 2, dtype=torch.float32) + 0.05
    maxs = mins + sizes
    boxes = torch.cat([mins, maxs], dim=-1)
    return boxes.to(dtype=dtype)


def _make_tensor(spec):
    dtype = _DTYPE_MAP[spec["dtype"]]
    shape = spec["shape"]
    name = spec["name"]
    value_range = spec.get("range")

    if dtype == torch.bool:
        return torch.randint(0, 2, tuple(shape), dtype=torch.int64).to(torch.bool)

    if name in {"boxes", "bboxes", "gtboxes"} and shape and shape[-1] == 4 and dtype in {
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    }:
        return _make_boxes(shape, dtype)

    if value_range is not None:
        low, high = value_range
        if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            high_exclusive = high + 1
            return torch.randint(low, high_exclusive, tuple(shape), dtype=dtype)
        return torch.empty(tuple(shape), dtype=dtype).uniform_(low, high)

    if dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        return torch.randint(0, 17, tuple(shape), dtype=dtype)

    return torch.randn(*shape, dtype=dtype)


def _make_tensor_list(spec):
    dtype = _DTYPE_MAP[spec["dtype"]]
    return [torch.randn(*shape, dtype=dtype) for shape in spec["shapes"]]


def _make_arg(spec):
    spec_type = spec["type"]
    if spec_type == "tensor":
        return _make_tensor(spec)
    if spec_type == "tensor_list":
        return _make_tensor_list(spec)
    if spec_type == "attr":
        return spec["value"]
    raise ValueError(f"Unsupported input spec type: {spec_type}")


def get_input_groups():
    input_groups = []
    for case in INPUT_CASES:
        input_groups.append([_make_arg(spec) for spec in case["inputs"]])
    return input_groups


def get_init_inputs():
    return []
