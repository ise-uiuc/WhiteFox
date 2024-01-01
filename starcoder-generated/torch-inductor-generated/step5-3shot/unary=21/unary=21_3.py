
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(256, 256, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(256, 256, 1, stride=1, padding=0)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x0):
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.relu(x2)
        return x3
# Inputs to the model
x0 = torch.randn(1, 256, 160, 320)
