
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, 1)
        self.conv2 = torch.nn.Conv2d(2, 4, 1)
        self.conv3 = torch.nn.Conv2d(4, 1, 1)
    def forward(self, x1, x2, x3, x4):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        y1 = torch.relu(torch.cat([x4, x1], 1))
        y2 = torch.relu(torch.cat([x3, y1], 1))
        y3 = torch.cat([x2, y2], 1)
        y4 = torch.cat([x1, y3], 1)
        y5 = torch.relu(y4)
        y6 = torch.cat([y3, y5], 1)
        y7 = torch.relu(y6)
        y8 = torch.cat([y2, y7], 1)
        y9 = torch.relu(y8)
        y10 = torch.cat([x3, y5], 1)
        y11 = torch.cat([x1, y10], 1)
        y12 = torch.relu(y11)
        y13 = torch.cat([x4, y12], 1)
        y14 = torch.relu(torch.cat([x3, y13], 1))
        y15 = torch.cat([x2, y14], 1)
        return y6

# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
x2 = torch.randn(1, 1, 1, 1)
x3 = torch.randn(1, 1, 1, 1)
x4 = torch.randn(1, 1, 1, 1)
