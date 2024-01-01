
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 16, 1, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.conv(x)
        v1 = self.relu(v1)
        v2 = v1 - 1
        return v2
# Inputs to the model
x = torch.randn(1, 1, 10, 10)
