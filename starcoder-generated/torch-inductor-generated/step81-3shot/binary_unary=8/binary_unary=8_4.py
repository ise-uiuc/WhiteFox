
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(1, 4, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = self.relu1(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
