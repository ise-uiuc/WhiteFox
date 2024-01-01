
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.nn.functional.conv2d(x, weight=0.567, bias=0.567)
        v2 = v1 - 0.356255
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
