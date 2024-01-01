
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        y = self.relu(self.conv1(x))
        z = torch.cat([y, y], dim=1)
        return z
# Inputs to the model
x = torch.randn(1, 1, 2, 2)
