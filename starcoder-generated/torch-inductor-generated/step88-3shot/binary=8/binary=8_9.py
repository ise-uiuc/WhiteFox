
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 15, stride=1, padding=7)
        self.conv2 = torch.nn.Conv2d(3, 8, 16, stride=1, padding=8)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2 
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 24, 24)
x2 = torch.randn(1, 3, 24, 24)
