
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, groups=192)
        self.conv1_2 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, groups=192)
        self.conv1_3 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, groups=192)
        self.conv1_4 = torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, groups=192)
    def forward(self, x1):
        x2 = self.conv1_1(x1)
        x3 = torch.tanh(x2)
        x4 = self.conv1_2(x3)
        x5 = torch.tanh(x4)
        x6 = self.conv1_3(x5)
        x7 = torch.tanh(x6)
        x8 = self.conv1_4(x7)
        x9 = torch.tanh(x8)
        return x9
# Inputs to the model
x1 = torch.rand(1, 192, 28, 28)
