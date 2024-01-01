
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv3_1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1[0])
        v2 = self.conv1(x1[1])
        v3 = v1 + v2
        v4 = self.conv3_1(x1[2])
        v5 = v1 + v4
        v6 = self.conv2(x1[0])
        v7 = self.conv2(x1[1])
        v8 = v6 + v7
        v9 = v5 + v8
        return v9
# Inputs to the model
x1 = [(torch.randn(1, 3, 32, 32)),
      (torch.randn(1, 3, 32, 32)),  
      (torch.randn(1, 3, 32, 32))]
