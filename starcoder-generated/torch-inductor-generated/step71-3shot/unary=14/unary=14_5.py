
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(48, 48, 2, stride=1, padding=1)
        self.relu = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 48, 32, 32)
