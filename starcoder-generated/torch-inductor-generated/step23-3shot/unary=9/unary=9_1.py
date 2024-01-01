
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        print("Before pointwise conv...")
        v1 = self.conv(x1)
        print("After pointwise conv...")
        v2 = 3 + v1
        print("After conv+const...")
        v3 = v2.clamp(min=0, max=6)
        print("After min/max CLAMP...")
        v4 = v3 / 6
        print("After DIV...")
        v5 = self.conv1(v4)
        print("Before pointwise conv...")
        v6 = 3 + v5
        print("After conv+const...")
        v7 = v6.clamp(min=0, max=6)
        print("After min/max CLAMP...")
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
