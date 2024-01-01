
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.qfc = torch.nn.Linear(256, 256, bias=False)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.relu(x1)
        v2 = self.qfc(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = -0.8
max = -0.3
# Inputs to the model
x1 = torch.randn(1, 256)
