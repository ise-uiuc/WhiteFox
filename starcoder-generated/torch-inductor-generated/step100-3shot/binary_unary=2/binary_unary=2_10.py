
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu(v1)
        v3 = v2 - torch.full((1, 1, 8, 8), 0.0, dtype=torch.float)
        v4 = self.relu(v3)
        v5 = v4 - torch.full((1, 1, 8, 8), -0.1, dtype=torch.float)
        v6 = self.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 8, 8)
