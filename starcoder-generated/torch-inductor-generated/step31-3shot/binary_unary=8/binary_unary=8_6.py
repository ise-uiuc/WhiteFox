
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v1 = torch.randn(16, 2, 3, 3)
    def forward(self, x1):
        v2 = self.v1
        v3 = torch.nn.functional.conv2d(x1, v2, padding=1, stride=1)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
