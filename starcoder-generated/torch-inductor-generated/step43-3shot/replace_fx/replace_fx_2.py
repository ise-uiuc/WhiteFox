
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3)
        self.dense1 = torch.nn.Linear(20, 20)
        self.dense2 = torch.nn.Linear(20, 20)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.relu1(x2)
        x4 = F.max_pool2d(x3, 2, 2)
        x5 = torch.flatten(x4, 1)
        x6 = self.dense1(x5)
        x7 = self.relu2(x6)
        x8 = self.dense2(x5)
        x9 = self.relu2(x8)
        x10 = torch.rand_like(x8)
        return x7
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
