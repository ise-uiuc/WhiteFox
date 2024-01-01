
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(16, 32, 3, padding=1, stride=2)
        self.linear = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.linear(v1)
        v3 = self.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 78, 78)
