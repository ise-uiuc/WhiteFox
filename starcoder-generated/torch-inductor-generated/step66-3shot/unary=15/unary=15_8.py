
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
        self.maxpool1 = torch.nn.MaxPool2d(3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.maxpool1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
