import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Fused FFT size padding and real FFT computation for Hyena convolution.
    Pads input to 2*seqlen for circular convolution, computes real FFT (rfft),
    and normalizes by fft_size.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Applies FFT size padding and real FFT computation for Hyena convolution.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, channels, seqlen].
                              Supports float32.

        Returns:
            tuple: (x_freq_real, x_freq_imag)
                - x_freq_real (torch.Tensor): Real part of normalized frequency domain output
                                              with shape [batch_size, channels, seqlen+1].
                - x_freq_imag (torch.Tensor): Imaginary part of normalized frequency domain output
                                              with shape [batch_size, channels, seqlen+1].
        """
        batch, channels, seqlen = x.shape
        fft_size = 2 * seqlen

        x_f32 = x.to(torch.float32)

        x_freq = torch.fft.rfft(x_f32, n=fft_size)

        x_freq = x_freq / fft_size

        x_freq_real = x_freq.real.contiguous()
        x_freq_imag = x_freq.imag.contiguous()

        return x_freq_real, x_freq_imag
