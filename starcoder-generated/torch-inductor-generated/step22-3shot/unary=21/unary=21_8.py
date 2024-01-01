
class tanhActivation(torch.nn.Module):
    def forward(self, x1):
        result = torch.tanh(x1)
        y = torch.add(x1, result)
        return result
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d = torch.nn.BatchNorm2d(16)
        self.tanh = tanhActivation()
        self.conv_2 = torch.nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d_1 = torch.nn.BatchNorm2d(16)
        self.tanh_1 = tanhActivation()
        self.conv_3 = torch.nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    def forward(self, x):
        v1 = self.conv_1(x)
        v2 = self.batchnorm2d(v1)
        v3 = self.tanh(v2)
        v4 = self.conv_2(v3)
        v5 = self.batchnorm2d_1(v4)
        v6 = self.tanh_1(v5)
        v7 = self.conv_3(v6)
        return v7
# Inputs to the model
x = torch.randn(1, 3, 256, 256)
