
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 11, stride=1, padding=3)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = F.relu(x2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 3, 100, 100)
