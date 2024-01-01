
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.batchNorm = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv2(x)
        v2 = torch.nn.functional.relu(v1)
        v3 = self.batchNorm(v2)
        v4 = torch.nn.functional.max_pool2d(v3, 2, stride=1)
        v5 = self.conv3(x)
        v6 = torch.nn.functional.relu(v5)
        v7 = self.batchNorm(v6)
        v8 = torch.nn.functional.max_pool2d(v7, 2, stride=1)
        v9 = v5 + v7
        v10 = self.conv4(v9)
        v11 = torch.nn.functional.relu(v10)
        v12 = self.batchNorm(v11)
        v13 = torch.nn.functional.max_pool2d(v12, 2, stride=1)
        v14 = v2 + v13
        v15 = v14 > 0
        v16 = v14 * -0.1
        v17 = torch.where(v15, v14, v16)
        return v17
# Inputs to the model
x1 = torch.rand(1, 64, 64, 64)
