
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, 1, stride=(1, 2), padding=(1, 2), bias=False)
        self.conv2 = torch.nn.Conv2d(8, 16, 5, stride=(2, 1), padding=(3, 3), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(16)

    def forward(self, x_0):
        x_1 = self.conv1(x_0)
        x_1 = torch.tanh(x_1)
        x_1 = self.bn1(x_1)
        x_1 = torch.tanh(x_1)
        x_2 = self.conv2(x_1)
        x_2 = torch.tanh(x_2)
        x_2 = self.bn2(x_2)
        x_2 = torch.tanh(x_2)
        return x_2
# Inputs to the model
x = torch.randn(1, 4, 4, 4)
