
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=12, out_channels=104, kernel_size=(1, 11), stride=(1, 1), padding=(0, 6))
        self.conv2 = torch.nn.Conv2d(104, 4, kernel_size=(1, 7), stride=(1, 1), padding=(0, 2))
        self.conv3 = torch.nn.ConvTranspose2d(4, 2, kernel_size=(1, 7), stride=(1, 1), padding=(0, 2))
        self.conv4 = torch.nn.Conv2d(2, 250, kernel_size=(14, 1), stride=(1, 1), padding=(0, 0))
        self.conv5 = torch.nn.Conv2d(250, 168, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv6 = torch.nn.Conv2d(168, 250, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
        self.conv7 = torch.nn.Conv2d(250, 510, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv8 = torch.nn.Conv2d(510, 129, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.nn.functional.avg_pool2d(v2, (1, 3), stride=(1, 3), padding=(0, 1))
        v4 = self.conv2(v3)
        v5 = torch.sigmoid(v4)
        v6 = self.conv3(v5)
        v7 = torch.sigmoid(v6)
        v8 = torch.nn.functional.avg_pool2d(v7, (15, 2), stride=(15,2), padding=(0, 0))
        v9 = self.conv4(v8)
        v10 = torch.functional.max_pool2d(v9, (3, 1), stride=(3, 1), padding=(1, 0))
        v11 = self.conv5(v10)
        v12 = torch.functional.max_pool2d(v11, (1, 1), stride=(1, 1), padding=(0, 0))
        v13 = torch.functional.max_pool2d(v12, (1, 1), stride=(1, 1), padding=(0, 0))
        v14 = self.conv6(v13)
        v15 = torch.functional.max_pool2d(v14, (1, 1), stride=(1, 1), padding=(0, 0))
        v16 = self.conv7(v15)
        v17 = torch.functional.max_pool2d(v16, (1, 1), stride=(1, 1), padding=(0, 0))
        v18 = self.conv8(v17)
        return v18
# Inputs to the model
x1 = torch.randn(1, 12, 15, 80)
