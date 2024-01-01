
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.conv2d(input=x1, weight=torch.ones(16, 3, 2, 2), stride=2, padding=(1, 1), dilation=1, groups=1)
        v2 = v1 + 1
        v3 = torch.clamp_max(v2, 1)
        v4 = v1 * v3
        v5 = v4 / 1
        return v5.flatten(1)
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
