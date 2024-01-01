
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1, stride=2)
        self.conv2 = torch.nn.Conv2d(3, 64, 3, padding=1, stride=2)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = torch.nn.functional.relu(v1 + v2)
        v4 = torch.nn.functional.tanh(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
x2 = torch.randn(1, 3, 128, 128)
