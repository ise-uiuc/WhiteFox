
class Model(torch.nn.Module):
        def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 4, 1, stride=1, padding=1)
        def forward(self, x1):
                return torch.relu(torch.add(self.conv1(x1), self.conv1(x1)))
# Inputs to the model
x1 = torch.randn(2, 1, 64, 64)
