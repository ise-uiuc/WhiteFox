
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu(v1)
        v3 = self.conv2(v2)
        v4 = self.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 100, 100)
