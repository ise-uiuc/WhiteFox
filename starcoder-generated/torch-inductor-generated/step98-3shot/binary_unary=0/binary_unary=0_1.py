
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 6, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(6, 3, 7, stride=1, padding=3)
    def forward(self, x1):
        a1 = x1[:, :6, :, :]
        v1 = self.conv1(a1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 9, 64, 64)
