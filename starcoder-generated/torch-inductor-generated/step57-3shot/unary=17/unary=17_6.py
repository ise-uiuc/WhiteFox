
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding_mode='zeros')
        self.conv2 = torch.nn.ConvTranspose2d(8, 32, 3, stride=(2, 2), padding=(4, 4), output_padding=(1, 1))
        self.conv3 = torch.nn.ConvTranspose2d(32, 64, 3, stride=(2, 2), padding=(4, 4), output_padding=(1, 1))
        self.conv4 = torch.nn.ConvTranspose2d(64, 128, 3, padding_mode='zeros')
        self.conv5 = torch.nn.ConvTranspose2d(128, 64, 3, padding_mode='zeros')
        self.conv6 = torch.nn.ConvTranspose2d(64, 32, 3, padding_mode='zeros')
        self.conv7 = torch.nn.ConvTranspose2d(32, 8, 3, padding_mode='zeros')
        self.pool = torch.nn.MaxPool2d(5, stride=(2, 2), padding=1)
    def forward(self, x0):
        v0 = self.pool(x0)
        v1 = self.conv1(v0)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        v7 = self.conv7(v6)
        v8 = self.pool(v7)
        v9 = self.conv1(v8)
        v10 = self.conv2(v9)
        v11 = self.conv3(v10)
        v12 = self.conv4(v11)
        v13 = self.conv5(v12)
        v14 = self.conv6(v13)
        v15 = self.conv7(v14)
        v16 = torch.mul(v15, v1)
        return v16
# Inputs to the model
x0 = torch.randn(1, 3, 41, 41)
