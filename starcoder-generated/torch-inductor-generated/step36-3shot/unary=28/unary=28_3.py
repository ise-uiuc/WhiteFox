
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2048, 1000, bias=True)
 
    def forward(self, x1, x2, x3, x4):
        v1 = self.linear(x1)
        k1 = torch.randint(-7, 7, (1, ))
        k2 = torch.randint(-128, 128, (1, ))
        v2 = torch.clamp_min(v1, k1)
        v3 = torch.clamp_max(v2, k2)
        k3 = torch.randint(-128, 128, (1, ))
        k4 = torch.randint(-128, 128, (1, ))
        v4 = torch.clamp_min(v3, k3)
        v5 = torch.relu6(v4, k4)
        k5 = torch.randint(-128, 128, (1, ))
        k6 = torch.randint(-128, 128, (1, ))
        v6 = torch.clamp_min(v5, k5)
        v7 = torch.relu6(v6, k6)
        k7 = torch.randint(-128, 128, (1, ))
        k8 = torch.randint(-128, 128, (1, ))
        v8 = torch.clamp_min(v7, k7)
        v9 = torch.relu6(v8, k8)
        v10 = v9 * x2
        v11 = x3 * v10 + x4
        return v11

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2048)
x2 = torch.randn(1, 1000)
x3 = torch.randn(1)
x4 = torch.randn(1, 1000)
