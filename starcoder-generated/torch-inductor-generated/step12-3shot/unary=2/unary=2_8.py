
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_0 = torch.nn.Conv2d(28, 4, (6, 5), (10, 11))
        self.relu_1 = torch.nn.ReLU(False)
        self.conv2d_2 = torch.nn.Conv2d(4, 1, (2, 1))
        self.conv2d_3 = torch.nn.Conv2d(22, 4, (8, 5), (8, 5))
        self.linear = torch.nn.Linear(192, 4)
        self.relu_9 = torch.nn.ReLU(False)
        self.conv2d_11 = torch.nn.Conv2d(4, 13, (23, 18), (11, 22))
        self.relu_13 = torch.nn.ReLU(False)
    def forward(self, x1, x2, x3):
        v1 = self.conv2d_0(x1)
        v2 = self.relu_1(v1)
        v3 = self.conv2d_2(v2)
        v4 = self.conv2d_3(x2)
        v5 = torch.flatten(v1, 1, -1)
        v6 = self.linear(v5)
        v7 = self.relu_9(v6)
        v8 = v7.view((v7.size()[0], 12, -1))
        v9 = v8 + x3
        v10 = self.conv2d_11(v9)
        v11 = self.relu_13(v10)
        return v2
# Inputs to the model
x1 = torch.randn(1, 28, 36, 13)
x2 = torch.randn(1, 22, 27, 12)
x3 = torch.randn(1, 13, 26, 133)
