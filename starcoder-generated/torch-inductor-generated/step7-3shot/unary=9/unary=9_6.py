
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x2):
        input = torch.Tensor(x2)
        v1 = self.conv(input)
        v2 = torch.add(v1, 3)
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = torch.true_divide(v4, 6)
        return v1, v2, v3, v4, v5
# Input to the model
x2 = torch.randn(1, 3, 64, 64)
