
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv3d(64, 64, 3, stride=1, padding=1, use_bias=True)
        self.conv4 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 256, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3, x4):
        v1 = x3.view(1, 64, 2, 49, 29)
        v2 = self.conv1(x4)
        v3 = x1.view(1, 32, 49, 29)
        v4 = v2.view(1, 64, 3, 64, 49, 29)
        v5 = x2.add(v1)
        v6 = self.conv5(v5)
        v7 = torch.reshape(v1, (-1, 2, 200))
        v8 = v6[:, :, :200]
        v9 = self.conv2(v7)
        v10 = self.conv3(v4))
        v11 = self.conv4(v8)
        v12 = torch.reshape(v10, (1, 4, 49, 29))
        return v12
# Inputs to the model
x1 = torch.randn(1, 16, 128, 192)
x2 = torch.randn(1, 32, 128, 192)
x3 = torch.randn(1, 64, 256, 256, 256)
x4 = torch.randn(1, 256, 128, 192)
