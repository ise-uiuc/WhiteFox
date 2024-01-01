
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 15), stride=(2, 1), padding=(3, 8))
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=(3, 11), stride=(2, 7), padding=(3, 6))
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=(1, 11), stride=(2, 7), padding=(3, 6))
        self.conv4 = torch.nn.Conv2d(128, 32, kernel_size=(3, 11), stride=(2, 7), padding=(3, 6))
        self.conv5 = torch.nn.Conv2d(4, 64, kernel_size=(4, 3), stride=(2, 2), padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv4(v6)
        v8 = torch.sigmoid(v7)
        v9 = self.conv5(x1)
        v10 = torch.sigmoid(v9)
        v11 = torch.cat([v8, v10], dim=1)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 4, 32, 32)
