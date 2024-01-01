
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 2, stride=1, padding=0, bias=True)
        self.conv2 = torch.nn.Conv1d(4, 1, 3, stride=1, padding=1, bias=True)
        self.conv3 = torch.nn.ConvTranspose2d(3, 1, 1, stride=1, padding=0, bias=True)
        self.conv4 = torch.nn.ConvTranspose1d(1, 4, 3, stride=1, padding=0, bias=True)
        self.leaky_relu = torch.nn.LeakyReLU(0.0001, inplace=True)
    def forward(self, x1):
        v1 = x1.reshape(1, 1, 3, 3)
        v2 = self.conv1(v1)
        v3 = v2.reshape(1, 4, 2, 2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
        v6 = v5.reshape(1, 4, 2, 2)
        v7 = self.leaky_relu(v6)
        v8 = self.conv2(v7)
        v9 = self.leaky_relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
