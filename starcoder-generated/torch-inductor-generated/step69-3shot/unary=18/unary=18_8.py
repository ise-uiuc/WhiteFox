
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3)
        self.conv2 = torch.nn.Conv2d(8, 4, 3)
        self.conv3 = torch.nn.Conv2d(4, 4, 3)
        self.max_pool = torch.nn.MaxPool2d(3)
        self.flatten = torch.nn.Flatten(1)
        self.linear1 = torch.nn.Linear(8, 1)
    def forward(self, x1):
        v1 = self.max_pool(self.conv1(x1))
        v2 = self.max_pool(self.conv2(v1))
        v3 = self.max_pool(self.conv3(v2))
        v4 = torch.flatten(v3, 1)
        v5 = self.linear1(v4)
        return nn.Sigmoid()(v5)
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
