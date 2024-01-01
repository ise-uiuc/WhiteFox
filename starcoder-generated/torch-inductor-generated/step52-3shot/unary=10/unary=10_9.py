
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8, bias= False)
        self.bn = torch.nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4/6
        out = self.bn(v5)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 16, 32, 32)
