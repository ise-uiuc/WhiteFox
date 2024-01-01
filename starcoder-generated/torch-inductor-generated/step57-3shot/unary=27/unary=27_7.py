
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.min
        v2 = self.max
        return v2
min = -2.23
max = -5.9
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
