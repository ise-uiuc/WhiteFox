
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(23, 10, 3, stride=2, padding=0, output_padding=0, bias=False)
        self.quantize_per_tensor = torch.quantization.QuantStub()
        self.quantize_per_channel_affine = torch.nn.quantized.Quantize(
            scale=0.0797954994239325,
            zero_point=0,
            axis=-1)
    def forward(self, x1):
        v1 = self.quantize_per_tensor(x1)
        v2 = self.conv_transpose(v1)
        v3 = v2 * 0.7978845608028654
        v4 = v3.dequantize()
        v5 = torch.tanh(v4)
        v6 = v5 * 1.0471975511965976
        v7 = v6.dequantize()
        v8 = self.quantize_per_channel_affine(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 23, 20, 4096)
