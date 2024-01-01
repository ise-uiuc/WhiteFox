
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 36, stride=1, padding=42)
        self.conv2 = torch.nn.Conv2d(3, 7, (11, 42), stride=1)
    def forward(self, x1):
       v1 = self.conv1(x1)
       v2 = v1 * 0.5
       v3 = v1 * 0.7071067811865476
       v4 = torch.erf(v3)
       v5 = v4 + 1
       v6 = v2 * v5
       return self.conv2(v6)
# Inputs to the model
x1 = torch.randn(1, 3, 11, 11)
