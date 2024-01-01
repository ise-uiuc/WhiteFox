
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.conv2d = torch.nn.Conv2d(64, 256, 1, 1)
        self.conv2d_1 = torch.nn.Conv2d(64, 128, 1, 1)
        self.conv2d_2 = torch.nn.Conv2d(128, 128, 3, 2, 1)
        self.conv2d_3 = torch.nn.Conv2d(128, 256, 1, 1)
    def forward(self, x):
        v1 = self.relu6(x)
        v2 = self.conv2d(v1)
        v3 = self.relu6(v2)
        v4 = self.conv2d_1(v3)
        v5 = self.relu6(v4)
        v6 = (v5,)
        v7 = self.conv2d_2(*v6)
        v8 = self.relu6(v7)
        v9 = self.conv2d_3(v8)
        v10 = self.relu6(v9)
        v11 = v10 + v10
        v12 = self.conv2d_1(v11)
        v13 = self.relu6(v12)
        return v13
# Inputs to the model
x = torch.randn(1, 64, 224, 224)
