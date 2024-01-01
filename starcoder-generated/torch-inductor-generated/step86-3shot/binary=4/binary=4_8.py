
```
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.bn = torch.nn.BatchNorm2d(8, affine=False)
        self.linear = torch.nn.Linear(8, 8, bias=False)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 1
        v3 = self.bn(v2)
        v4 = self.bn(v3)
        v5 = self.bn(v4)
        v6 = self.linear(v5)
        v7 = self.linear(v6)
        return v7
```

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
