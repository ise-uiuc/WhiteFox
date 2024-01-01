
class Model(torch.nn.Module):
    def __init__(self, a0):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(a0, 3, 5, stride=2, padding=2, output_padding=1)
        self.conv1 = torch.nn.AvgPool2d((2, 2), stride=(2, 1))
        self.conv2 = torch.nn.ConvTranspose2d(12, 3, 5, stride=2, padding=3, output_padding=2)
        self.conv3 = torch.nn.AvgPool2d(2, stride=(2, 1))
        self.conv4 = torch.nn.Conv1d(16, 9, 3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = v3.expand((-1, 3, -1, -1))
        v5 = torch.cat([v4, x1], dim=1)
        v6 = self.conv2(v5)
        v7 = torch.cat([x1, v6], dim=1)
        v8 = self.conv3(v7)
        v9 = self.conv4(v8)
        v10 = torch.relu(v9)
        v11 = torch.sigmoid(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
