
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        negative_slope = 0.2
        v1 = self.relu(x)
        v2 = v1 * negative_slope
        return v2
# Inputs to the model
x1 = torch.randn(4, 4, 64, 64)
