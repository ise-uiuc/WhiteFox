
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        a = 1.025
        b = -9.8732
        v2 = torch.sub(v1, torch.scalar_tensor(a, dtype=torch.float32))
        v3 = torch.sub(v2, b)
        return v3
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
