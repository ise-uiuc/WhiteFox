
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = torch.randn(1, 8, 1, 1)
        v2 = torch.randn(1, 8, 1, 1)
        v3 = v1 + v2
        v4 = v3 + x1
        v5 = v3 + x2
        return v4.div(v5)
# Inputs to the model
x1 = torch.randn(1, 3, 1, 1)
x2 = torch.randn(1, 3, 1, 1)
