
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(6, 16, 5, stride=1, padding=1)
        self.flatten = torch.nn.Flatten()
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.relu(v1)
        v3 = torch.nn.functional.relu(v1)
        v4 = self.conv2(v1)
        v5 = torch.nn.functional.max_pool2d(v3, 2)
        v6 = self.conv2(v5)
        v7 = torch.nn.functional.max_pool2d(v3, 2)
        v8 = torch.flatten(v6, 1)
        v9 = torch.flatten(v7, 1)
        v10 = torch.add(v8, v9)
        return v10
# Inputs to the model
x = torch.randn(1, 3, 1, 1)
