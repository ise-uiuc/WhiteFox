
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        r1 = torch.relu(torch.clamp(torch.max(torch.add(self.conv(x1), 3), torch.Tensor([0])), 0, 6))
        v1 = r1 / 6
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
