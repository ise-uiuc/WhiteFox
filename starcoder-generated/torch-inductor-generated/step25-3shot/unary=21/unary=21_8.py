
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # The Tanh layer uses the layer-wise quantization formula to quantize the output tensor of the pointwise convolution into 6 bits.
        self.l1 = torch.nn.Conv2d(3, 3, 2, stride=1, padding=0)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.l1(x)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(30, 3, 128, 128)
