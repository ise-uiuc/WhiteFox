
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v = torch.zeros_like(x1)
        v = self.conv(x1)
        v = torch.nn.functional.relu(v)
        return v
# Inputs to the model
x1 = torch.randn(1, 3, 32, 64)
