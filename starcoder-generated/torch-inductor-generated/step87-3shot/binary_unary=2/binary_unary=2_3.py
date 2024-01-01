
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1 - 0.1)
        v3 = torch.pow(v2, 2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 56, 56)
