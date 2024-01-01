
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.conv_transpose2d(input=x1, weight=torch.randn(5, 5, 2, 2), bias=None, stride=(3, 3), padding=(0, 0), output_padding=(1, 0), groups=5, dilation=1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
