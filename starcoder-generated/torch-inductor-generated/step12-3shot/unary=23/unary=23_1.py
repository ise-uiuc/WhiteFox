
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 4, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.nn.LayerNorm(8, eps=1.0e-05, elementwise_affine=True)(v1)
        v3 = v2 + v1
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
