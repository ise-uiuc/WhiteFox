
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(2, 2, (3, 3), padding=(1, 1), stride=(1, 1), dilation=(1, 1))
        self.conv2 = torch.nn.ConvTranspose2d(2, 2, (3, 3), padding=(2, 2), stride=(1, 1), dilation=(2, 2), groups=2)
        self.conv3 = torch.nn.ConvTranspose2d(2, 2, (3, 3), padding=(4, 4), stride=(1, 1), dilation=(4, 4), groups=2)
        self.conv4 = torch.nn.ConvTranspose2d(2, 2, (3, 3), padding=(8, 8), stride=(1, 1), dilation=(8, 8), groups=2)
        self.conv5 = torch.nn.ConvTranspose2d(2, 2, (3, 3), padding=(1, 1), stride=(1, 1), dilation=(1, 2))
        self.conv6 = torch.nn.ConvTranspose2d(2, 2, (3, 3), padding=(1, 1), stride=(1, 1), dilation=(1, 4))
        self.conv7 = torch.nn.ConvTranspose2d(2, 2, (3, 3), padding=(2, 2), stride=(1, 1), dilation=(2, 2))
        self.conv8 = torch.nn.ConvTranspose2d(2, 2, (3, 3), padding=(4, 4), stride=(1, 1), dilation=(4, 4), groups=2)
        self.conv9 = torch.nn.ConvTranspose2d(2, 2, (3, 3), padding=(2, 2), stride=(1, 1), dilation=(2, 1))
        self.conv10 = torch.nn.ConvTranspose2d(2, 2, (3, 3), padding=(1, 1), stride=(1, 1), groups=2)
        self.conv11 = torch.nn.ConvTranspose2d(2, 2, (3, 3), padding=(1, 1), stride=(1, 1), dilation=(1, 1), groups=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = self.conv4(x1)
        v5 = self.conv5(x1)
        v6 = self.conv6(x1)
        v7 = self.conv7(x1)
        v8 = self.conv8(x1)
        v9 = self.conv9(x1)
        v10 = self.conv10(x1)
        v11 = self.conv11(x1)
        v12 = torch.relu(v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10 + v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 2, 4, 4)
