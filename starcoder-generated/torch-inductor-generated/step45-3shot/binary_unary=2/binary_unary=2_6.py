
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()   
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = self.conv2(v2)
        v4 = v3 - 10
        v5 = F.relu(v4)
        v6 = self.conv1(v5)
        v7 = v6 - 11
        v8 = F.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
