
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(2, 512, 3, stride=(1, 1))
        self.conv2 = torch.nn.Conv3d(516, 384, 1, stride=(1, 1))
        self.conv3 = torch.nn.Conv3d(772, 384, 3, stride=(1, 1, 1), dilation=(1, 1, 1))
        self.conv4 = torch.nn.Conv3d(804, 384, 3, stride=(1, 1, 1), dilation=(3, 3, 3))
        self.conv5 = torch.nn.Conv3d(812, 384, 3, stride=(1, 1, 1), dilation=(5, 5, 5))
        self.conv6 = torch.nn.Conv3d(964, 64, 1, stride=(1, 1))
    def forward(self, x):
        y = self.conv1(x)
        y = torch.tanh(y)
        y = self.conv2(y)
        y = torch.tanh(y)
        y = self.conv3(y)
        y = torch.tanh(y)
        y = self.conv4(y)
        y = torch.tanh(y)
        y = self.conv5(y)
        y = torch.tanh(y)
        y = self.conv6(y)
        y = torch.tanh(y)
        return y
# Inputs to the model
x = torch.randn(1, 2, 300, 224, 224)
