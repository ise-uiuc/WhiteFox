
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 7)
        self.conv2 = torch.nn.Conv2d(8, 12, 7)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + x2
        v4 = torch.nn.functional.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(3, 3, 14, 14)
x2 = torch.randn(3, 3, 14, 14)
