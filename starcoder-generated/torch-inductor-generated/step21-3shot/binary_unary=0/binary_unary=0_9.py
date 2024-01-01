
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = torch.relu(x1)
        v2 = v1 + x1
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
