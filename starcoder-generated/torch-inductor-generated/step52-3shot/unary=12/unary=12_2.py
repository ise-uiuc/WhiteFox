
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 9, 3, stride=1, padding=0)
        self.linear = torch.nn.Linear(4, 8)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.linear(x1)
        v3 = v1 * v2
        v4 = self.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 7, 64, 64)
