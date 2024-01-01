
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = torch.nn.Conv2d(3, 42, 1, stride=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv2d1(x1)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(3, 3, 42, 42)
