
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 5, stride=1, padding=2)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu1(v1)
        v2 = v1 + v2
        v3 = self.conv2(v2)
        v3 = self.relu1(v3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 57, 57)
