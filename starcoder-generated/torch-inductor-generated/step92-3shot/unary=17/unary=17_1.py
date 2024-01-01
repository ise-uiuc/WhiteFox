
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.conv2 = torch.nn.ConvTranspose2d(64, 2, kernel_size=(5, 5), stride=(2, 2), bias=True)
        self.conv3 = torch.nn.Conv2d(2, 6, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.prelu_1 = torch.nn.PReLU()
        self.prelu_2 = torch.nn.PReLU()
        self.prelu_3 = torch.nn.PReLU()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.prelu_1(v1)
        v3 = self.conv2(v2)
        v4 = self.prelu_2(v3)
        v5 = self.conv3(v4)
        v6 = self.prelu_3(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
