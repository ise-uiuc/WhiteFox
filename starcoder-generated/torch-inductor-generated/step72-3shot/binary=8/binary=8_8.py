
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=1, stride=1, bias=False)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=1, stride=1, bias=False, padding=(0, 0))
        self.conv3 = torch.nn.Conv2d(64, 32, kernel_size=1, stride=1, bias=False)
        self.conv4 = torch.nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False)
        self.conv5 = torch.nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=False)
        self.conv6 = torch.nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False, padding=(0, 0))
    def forward(self, x1, x2, x3, x4):
        v9 = self.conv6(x3)
        v27 = self.conv1(x2)
        v17 = self.conv2(x1)
        v8 = self.conv3(x1)
        v24 = self.conv2(x4)
        v12 = self.conv4(x4)
        v29 = self.conv2(v9)
        v28 = self.conv2(x1)
        v21 = self.conv5(x2)
        v15 = self.conv5(x3)
        v26 = v17 + v24
        v16 = (v9 + v12) + v28
        v25 = v15 + v21
        v10 = (v26 + v8) + v27
        v19 = v10 + v29
        return v19
# Inputs to the model
x1 = torch.randn(1, 1, 128, 2048)
x2 = torch.randn(1, 32, 128, 2048)
x3 = torch.randn(1, 64, 128, 2048)
x4 = torch.randn(1, 3, 128, 2048)
