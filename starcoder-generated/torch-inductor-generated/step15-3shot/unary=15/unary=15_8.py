
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.relu = torch.nn.functional.relu
        self.linear = torch.nn.Linear(100, 2)
    def forward(self, x1):
        v1 = torch.relu(self.conv(x1))
        v2 = torch.mean(v1, dim=(1, 2, 3))
        v3 = self.relu(self.linear(v1))
        return v3
# Inputs to the model
x1 = torch.randn(16, 3, 256, 256)
