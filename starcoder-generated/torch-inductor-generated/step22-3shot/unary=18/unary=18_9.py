
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=218, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv2 = torch.nn.Conv2d(218, 12, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv3 = torch.nn.Conv2d(12, 218, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv4 = torch.nn.Conv2d(218, 12, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv5 = torch.nn.Conv2d(12, 12, kernel_size=(1, 1), stride=(1, 1), padding=(0, 1))
        self.conv6 = torch.nn.Conv2d(12, 650, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv7 = torch.nn.Conv2d(650, 160, kernel_size=(12, 1), stride=(1, 1), padding=0)
        self.conv8 = torch.nn.Conv2d(160, 2, kernel_size=(1, 1), stride=(1, 1), padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.relu(v2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
        v6 = torch.relu(v5)
        v7 = self.conv5(v6)
        v8 = torch.relu(v7)
        v9 = self.conv6(v8)
        v10 = self.conv7(v9)
        v11 = torch.sigmoid(v10)
        v12 = self.conv8(v11)
        return v12
# Inputs to the model
x1 = torch.randn(5, 3, 93, 165)
