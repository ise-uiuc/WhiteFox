
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.ConvTranspose2d(128, 24, 3)
        self.conv2_1 = torch.nn.ConvTranspose2d(24, 32, 3)
        self.conv2_2 = torch.nn.ConvTranspose2d(32, 32, 3)
        self.conv2_3 = torch.nn.ConvTranspose2d(32, 24, 3)
        self.conv2_4 = torch.nn.ConvTranspose2d(24, 16, 3)
        self.conv2_5 = torch.nn.ConvTranspose2d(16, 2, 3)
        self.conv2_6 = torch.nn.ConvTranspose2d(2, 3, 3)
        self.conv2_7 = torch.nn.ConvTranspose2d(3, 1, 3)
    def forward(self, x):
        v0 = self.conv2(x)
        v1 = torch.tanh(v0)
        v2 = self.conv2_1(v1)
        v3 = torch.tanh(v2)
        v4 = self.conv2_2(v3)
        v5 = torch.tanh(v4)
        v6 = self.conv2_3(v5)
        v7 = torch.tanh(v6)
        v8 = self.conv2_4(v7)
        v9 = torch.tanh(v8)
        v10 = self.conv2_5(v9)
        v11 = torch.tanh(v10)
        v12 = self.conv2_6(v11)
        v13 = torch.tanh(v12)
        v14 = self.conv2_7(v13)
        v15 = torch.tanh(v14)
        return v15
# Inputs to the model
x1 = torch.randn(1, 128, 128, 128)
