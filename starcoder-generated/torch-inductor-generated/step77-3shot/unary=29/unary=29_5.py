
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.5128, max_value=-1.4012):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x1):
        v1 = self.relu(x1)
        v2 = torch.clamp(v1, min=-0.3681, max=-0.3295)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 29, 27)
