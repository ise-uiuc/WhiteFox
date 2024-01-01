
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.conv2d(input=x1, weight=torch.nn.Parameter(torch.ones(5, 3, 3, 3)), bias=None, stride=(2, 2), padding=(0, 0), dilation=1, groups=1) * 0.5
        v2 = torch.conv2d(input=x1, weight=torch.nn.Parameter(torch.ones(5, 3, 3, 3)), bias=None, stride=(2, 2), padding=(0, 0), dilation=1, groups=1) * 0.7071067811865476
        v3 = torch.erf(input=v2)
        v4 = v3 + 1
        v5 = v1 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
