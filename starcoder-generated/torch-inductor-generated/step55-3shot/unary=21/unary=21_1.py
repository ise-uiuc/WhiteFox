
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, (1, 7), 2)
        self.conv2 = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=19, stride=3, bias=True, padding_mode='zeros')
        self.conv3 = torch.nn.Conv2d(32, 16, 3, 1, 2)
        self.conv4 = nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=1, stride=1, bias=True, padding_mode='zeros')
        self.conv5 = torch.nn.Conv3d(64, 32, 3, padding=1)
        self.conv6 = torch.nn.Conv1d(32, 16, 3, padding=1)
        self.conv7 = torch.nn.ConvTranspose2d(16, 16, 6, stride=(1, 1))
        self.conv8 = torch.nn.Conv1d(16, 8, 3, 1, 2)
        self.conv9 = torch.nn.ConvTranspose3d(64, 32, 6, (1, 1), (1, 1))
        self.conv10 = torch.nn.Conv2d(16, 8, 3, stride=2)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv3(v4)
        v6 = self.conv4(v5)
        v7 = torch.tanh(v6)
        v8 = self.conv5(v7)
        v9 = self.conv6(v8)
        v10 = torch.tanh(v9)
        v11 = self.conv7(v10)
        v12 = self.conv8(v11)
        v13 = torch.tanh(v12)
        v14 = self.conv9(v13)
        v15 = self.conv10(v14)
        v16 = torch.tanh(v15)
        return v16
# Inputs to the model
x = torch.randn(1, 16, 24, 24, 3)
