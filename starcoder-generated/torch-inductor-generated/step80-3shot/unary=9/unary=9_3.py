
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[[[1.0, 2, 3], [4, 5, 6], [7, 8, 9]]]])
        self.conv = torch.nn.Conv2d(3, 1, kernel.shape, stride=2, padding=1, bias=False)
        self.conv.weight = torch.nn.Parameter(kernel, requires_grad=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add(3)
        v3 = v2.clamp(min=0, max=6)
        v4 = v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
