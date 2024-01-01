
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 2, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(2, 4, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        return torch.nn.functional.relu(v2)
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
