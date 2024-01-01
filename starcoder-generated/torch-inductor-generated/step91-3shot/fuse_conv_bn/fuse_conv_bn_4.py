
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 6, 6, bias=True)
        self.bn1 = torch.nn.BatchNorm2d(6)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(6, 6, 18, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, input):
        x = self.relu(self.bn1(self.conv1(input)))
        y = self.relu(self.conv2(x))
        y = self.sigmoid(y)
        return y
# Inputs to the model
input = torch.randn(2, 6, 18, 18)
