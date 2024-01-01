
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        return torch.relu(self.conv1(x))
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
