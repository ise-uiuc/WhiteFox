
model1 = torch.nn.ConvTranspose2d(66, 33, 3, stride=2, padding=1)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = model1 # Reuse the pointwise transposed convolution layer
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 66, 32, 32)
