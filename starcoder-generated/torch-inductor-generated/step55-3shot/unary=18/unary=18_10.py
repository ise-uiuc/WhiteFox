
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(4, 4, kernel_size=1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv3d(4, 4, kernel_size=1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv3d(4, 4, kernel_size=1, stride=1, padding=1)
        self.conv5 = torch.nn.Conv1d(4, 4, kernel_size=3, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(4, 4, kernel_size=(2, 4), stride=1, padding=(0, 2))
        self.conv8 = torch.nn.Conv2d(4, 4, kernel_size=(3, 3), stride=1, padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v4)
        v7 = self.conv7(v4)
        v8 = self.conv8(v4)
        v9 = torch.sigmoid(v5)
        v10 = torch.sigmoid(v6)
        v11 = torch.sigmoid(v7)
        v12 = torch.sigmoid(v8)
        return v9, v10, v11, v12
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
