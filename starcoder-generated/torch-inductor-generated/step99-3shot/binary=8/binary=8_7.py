
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1_1(x1)
        v2 = self.conv1_2(x1)
        v3 = v1.add(v2)
        v4 = self.conv2_1(x2)
        v5 = self.conv2_2(x2)
        v6 = v4.add(v5)
        return (v3, v6)
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn(1, 3, 16, 16)
