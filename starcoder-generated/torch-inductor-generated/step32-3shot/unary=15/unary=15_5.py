
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(784, 140, 1, stride=1, padding=0)
        self.dense1 = torch.nn.Linear(140, 140)
        self.dense2 = torch.nn.Linear(140, 140)
        self.dense3 = torch.nn.Linear(140, 140)
        self.dense4 = torch.nn.Linear(140, 140)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = torch.nn.functional.avg_pool2d(v2, kernel_size=2, stride=2, padding=0)
        v4 = v3.view(-1, 140)
        v5 = self.dense1(v4)
        v6 = torch.relu(v5)
        v7 = self.dense2(v6)
        v8 = torch.relu(v7)
        v9 = self.dense3(v8)
        v10 = torch.relu(v9)
        v11 = self.dense4(v10)
        v12 = torch.relu(v11)
        return v12
# Inputs to the model
x1 = torch.randn(10, 784, 1, 1)
