
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6_1 = torch.nn.ReLU6(True)
    def forward(self, x):
        negative_slope = 0.55671875
        v1 = self.relu6_1(x)
        v2 = v1 * negative_slope
        return v2
# Inputs to the model
x1 = torch.randn(2, 12, 2, 18)
