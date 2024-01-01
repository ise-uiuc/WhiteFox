
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 253, stride=[2, 2], padding=[16, 16], dilation=[1, 1], groups=64, bias=True)
        self.tanh1 = torch.nn.Tanh()
        self.conv2 = torch.nn.Conv2d(64, 253, 1, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=253, bias=True)
        self.tanh2 = torch.nn.Tanh()
        self.conv3 = torch.nn.Conv2d(253, 253, 1, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=253, bias=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh1(x)
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.conv3(x)
        return x
# Inputs to the model
x = torch.randn(9, 64, 179, 141)
