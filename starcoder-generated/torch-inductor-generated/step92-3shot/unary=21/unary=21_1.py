
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 1)
    def forward(self, x):
        v = self.conv1(x)
        v1 = torch.tanh(v)
        for _ in range(2):
            v1 = self.conv1(v1)
            v2 = torch.tanh(v1)
            v = v2
        return v2
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
