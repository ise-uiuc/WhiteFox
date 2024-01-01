
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1, stride=1, padding=0, bias=False)
        self.conv2 = torch.nn.Conv2d(4, 8, 3, stride=1, padding=1, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 5
        v3 = self.conv2(v2)
        v4 = v3 - 12
        v5 = F.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
