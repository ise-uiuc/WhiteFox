
# Example 3: PyTorch model with a pointwise transposed convolution that multiplies input tensors by 1/2 plus 3
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 8, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.abs(v1 - 8)
        v3 = v2 + 3
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 15, 15)
