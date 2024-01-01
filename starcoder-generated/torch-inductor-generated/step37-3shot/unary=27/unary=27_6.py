
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.relu(x1)
        v2 = self.maxpool(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 0.4
max = 0.99
# Inputs to the model
x1 = torch.randn(1, 490, 365)
