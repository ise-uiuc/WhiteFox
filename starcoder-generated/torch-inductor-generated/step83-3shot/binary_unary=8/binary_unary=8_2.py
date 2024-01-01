
class Model(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
            self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        def forward(self, x):
            v1 = self.conv1(x)
            v2 = self.conv2(x)
            v3 = torch.relu(v1 + v2)
            return v3

    def __init__(self):
        super().__init__()
        self.block1 = self.Block()
        self.block2 = self.Block()
        self.block3 = self.Block()

    def forward(self, x):
        v1 = self.block1(x)
        v2 = self.block2(v1)
        v3 = self.block3(v2)
        v4 = self.block1(v3)
        v5 = self.block2(v4)
        v6 = self.block3(v5)
        v7 = torch.cat([v6, v3], dim=1)
        v8 = torch.relu(self.block1(v7))
        return v8
# Inputs to the model
x1 = torch.randn(1, 8, 32, 32)
