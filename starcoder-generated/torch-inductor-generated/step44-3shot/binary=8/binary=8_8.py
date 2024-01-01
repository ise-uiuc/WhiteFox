


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.linear1 = torch.nn.Linear(32 * 32, 32 * 32)
        self.linear2 = torch.nn.Linear(32 * 32, 32 * 32)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = torch.nn.functional.avg_pool2d(x2, kernel_size=3)
        v4 = self.linear1(v3)
        v5 = self.linear2(v3)
        v6 = v4 / v5
        v7 = v1.sub(v2)
        v8 = v6 + v7
        v9 = self.conv3(x1)
        v10 = self.conv4(x2)
        v11 = v9 * v10
        v12 = torch.nn.functional.adaptive_avg_pool2d(x2, output_size=1)
        v13 = self.linear2(v12)
        v14 = self.linear1(v12)
        v15 = v13 - v14
        v16 = v12.div(v15)
        return v8 * v11 * v16
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
