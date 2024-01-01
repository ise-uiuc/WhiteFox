
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(256, 128, 1, stride=1, padding=0)
        self.relu = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 256, 27, 27)
