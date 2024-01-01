
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 200, 7, stride=1, padding=3, dilation=1, groups=1)
        self.conv2 = torch.nn.Conv2d(200, 200, 1, stride=1, padding=0, dilation=1, groups=1)
        self.conv3 = torch.nn.Conv2d(200, 200, 5, stride=1, padding=2, dilation=1, groups=1)
        self.conv4 = torch.nn.Conv2d(200, 2, 1, stride=1, padding=0, dilation=1, groups=1)
        self.conv5 = torch.nn.Conv1d(32, 64, 3, stride=2, padding=(1, 1), dilation=1, groups=1, bias=True)
        self.conv6 = torch.nn.Conv2d(32, 96, 3, stride=(2, 1), padding=1, dilation=1, groups=1, bias=True)
        self.conv7 = torch.nn.Conv3d(32, 52, 3, stride=2, padding=(2, 0, 6), dilation=1, groups=1, bias=True)
        self.conv8 = torch.nn.Conv3d(32, 64, 3, stride=2, padding=(2, 0, 5), dilation=1, groups=1, bias=True)
    def forward(self, x3):
        v4 = self.conv5(x3)
        v5 = torch.tanh(v4)
        v7 = self.conv7(x3)
        v8 = torch.tanh(v7)
        v10 = self.conv3(v8)
        v10 = v10.permute(0, 2, 3, 1)
        v10 = self.conv2(v10)
        v11 = v10.permute(0, 3, 1, 2)
        v11 = torch.tanh(v11)
        v11 = v11.squeeze(1)
        v12 = self.conv1(v11)
        v13 = v12.permute(0, 2, 1)
        v13 = torch.tanh(v13)
        v14 = v13.squeeze(1)
        v15 = torch.zeros_like(v13)
        v15[:, -1, :] = v13[:, -1, :]
        v15[:, -2, :] = v13[:, -2, :]
        v15[:, 0, :] = v13[:, 0, :]
        v15[:, 1, :] = v13[:, 1, :]
        v16 = v12.permute(0, 2, 1)
        v16 = self.conv4(v16)
        v17 = v16.squeeze(1).permute(0, 2, 1)
        v18 = torch.tanh(v17)
        v19 = self.conv6(x3)
        v20 = torch.tanh(v19)
        v21 = self.conv8(x3)
        v22 = torch.tanh(v21)
        return v18, v20, v22
# Inputs to the model
x3 = torch.randn(3, 32, 224)
