
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 4, 3, stride=2, padding=1) 
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - torch.randn(3, 8, 1, 1)
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 3
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
