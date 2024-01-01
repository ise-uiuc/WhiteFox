
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv_1 = torch.nn.Conv2d(8, 32, 3, padding=1)
        self.conv_2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv_3 = torch.nn.Conv2d(64, 32, 3, padding=1)
        self.conv_4 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv_5 = torch.nn.Conv2d(64, 16, 3, padding=1)
        self.conv_6 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv_7 = torch.nn.Conv2d(32, 8, 3, padding=1)
        self.conv_8 = torch.nn.Conv2d(8, 1, 3, padding=1)

        self.pool_1 = torch.nn.MaxPool2d(2)
        self.pool_2 = torch.nn.MaxPool2d(2)
        self.pool_3 = torch.nn.MaxPool2d(2)
    def forward(self, x):
        x = x.float()
        x = self.pool_1(x)
        x = self.conv_1(x)
        x = torch.tanh(x)
        x = self.conv_2(x)
        x = torch.tanh(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = torch.tanh(x)
        x = self.pool_2(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = torch.tanh(x)
        x = self.conv_7(x)
        x = self.conv_8(x)
        return x
# Inputs to the model
x= torch.randn(1, 8, 140, 140)
