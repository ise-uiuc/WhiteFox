
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv1_3 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2_3 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv1_1(x1)
        v2 = self.conv1_2(x1)
        v3 = self.conv1_3(x3)
        v4 = v1 + v2 + v3
        v5 = self.conv2_1(x2)
        v6 = self.conv2_2(x2)
        v7 = self.conv2_3(x3)
        v8 = v5 + v6 + v7
        return (v4, v8)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
x3 = torch.randn(1, 3, 32, 32)
