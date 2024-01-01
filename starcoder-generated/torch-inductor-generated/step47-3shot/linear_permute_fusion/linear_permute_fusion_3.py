
class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1, 1, (1, 1))
        self.flatten1 = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4, bias=False)
        self.relu1 = torch.nn.ReLU()
        self.flatten2 = torch.nn.Flatten()
        self.linear3 = torch.nn.Linear(8, 8)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.flatten1(v1)
        v3 = self.linear1(v1)
        v4 = self.relu1(v2)
        v5 = self.linear2(v3)
        v6 = self.relu1(v4)
        v7 = torch.matmul(v5, v6)
        v8 = self.flatten2(v6)
        v9 = self.linear3(v8)
        return v7 + v8 + v9
# Inputs to the model
x1 = torch.randn(1, 1, 6, 6)
