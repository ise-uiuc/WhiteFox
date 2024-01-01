
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = F.relu(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 14, 14)
