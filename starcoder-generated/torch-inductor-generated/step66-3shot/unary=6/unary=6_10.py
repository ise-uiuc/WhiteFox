
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, bias=True) # Conv1 begins
        self.conv1b = torch.nn.Conv2d(8, 3, 3, stride=3, padding=0, bias=True) # Conv1b begins
        self.conv2 = torch.nn.Conv2d(8, 3, 3, stride=1, padding=3, bias=False) # Conv2 begins
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1b(v1)
        v3 = self.conv2(v2) # Conv2 ends, and then Conv1b ends
        v4 = 3 + v3
        v5 = torch.clamp(v4, 0, 6)
        v6 = v3 * v5
        v7 = v6/6
        return v7
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
