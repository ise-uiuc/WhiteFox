
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 256, 1, stride=1, padding=1)
        self.linear = torch.nn.Linear(64, 64)
        self.flatten = torch.nn.Flatten()
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.flatten(v1)
        v3 = self.linear(v2)
        v4 = self.relu(v3)
        v5 = v4 - 622.10
        return v5
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
