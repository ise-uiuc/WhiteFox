
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, kernel_size=(1, 1), stride=(1, 1))
        self.conv2 = torch.nn.Conv2d(5, 6, kernel_size=(3, 1), stride=(1, 1))
        self.conv3 = torch.nn.Conv2d(6, 5, kernel_size=(5, 1), stride=(1, 1))
    def forward(self, x):
        y = self.conv1(x)
        v0 = self.conv2(y)
        v1 = v0 - 0
        v2 = self.conv2(v1)
        v3 = v1 - 0
        v4 = self.conv3(v1)
        v5 = v4 - 0
        v6 = self.conv3(v3)
        v7 = v6 - 0
        v8 = self.conv2(v5)
        v9 = v8 - 0
        v10 = self.conv2(v7)
        v11 = v10 - 0
        v12 = self.conv2(v9)
        v13 = v12 - 0
        v14 = self.conv3(v11)
        v15 = v14 - 0
        return (v10)
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
