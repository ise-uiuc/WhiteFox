
class Model(torch.nn.Module):
    def __init__(self, activation, bias, min, max):
        super().__init__()
        self.min = min
        self.max = max
        self.act = getattr(torch, activation)
        self.fc1 = torch.nn.Linear(1200, 1600)
        self.fc2 = torch.nn.Linear(1600, 1600)
        self.bias = bias
    def forward(self, x1):
        v1 = self.act(self.fc1(x1))
        v1 = v1 + self.bias
        v1 = torch.clamp_min(v1, self.min)
        v1 = torch.clamp_max(v1, self.max)
        v1 = self.fc2(v1)
        v1 = v1 + self.bias
        v1 = torch.clamp_min(v1, self.min)
        v1 = torch.clamp_max(v1, self.max)
        return v1
activation = 'tanh'
bias = 5
min = 0.1
max = 50.3
# Inputs to the model
x1 = torch.randn(1, 1200)
