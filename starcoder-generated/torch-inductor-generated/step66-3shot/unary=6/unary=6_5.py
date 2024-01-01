
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 6 + v1
        v3 = self.relu(v2)
        v4 = v1 * v3
        v5 = v4.div(9)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 40, 40)
