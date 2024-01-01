
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128,2048, (1,1), stride=(1,1), groups=128)
        self.conv2 = torch.nn.Conv2d(2048,3072, (1,1), stride=(1,1), groups=2048)
        self.conv3 = torch.nn.Conv2d(3072,4928, (1,1), stride=(1,1), groups=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.add(v3, 3.14159)
        v5 = v2 + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 128, 35, 45)
