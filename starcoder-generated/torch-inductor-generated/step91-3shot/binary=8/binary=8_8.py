
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5, stride=1, padding=2)
        self.relu1 = torch.nn.ReLU()
    def forward(self, x1, x2):
        v1 = self.conv1(torch.add(x1, x2))
        v2 = v1 + v1
        v3 = self.relu1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
