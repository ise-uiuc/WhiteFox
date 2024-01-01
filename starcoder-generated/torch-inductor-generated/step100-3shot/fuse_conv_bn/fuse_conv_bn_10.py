
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = torch.nn.Conv2d(64, 32, 5)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(32, 16, 5)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(16, 16, 5)
        self.activation = torch.sigmoid
    def forward(self, x4):
        y1 = self.conv1(x4)
        y2 = self.relu1(y1)
        y3 = self.conv2(y2)
        y4 = self.relu2(y3)
        y5 = self.conv3(y4)
        y6 = self.activation(y5)
        return y6
# Inputs to the model
x4 = torch.randn(4, 64, 32, 32)
