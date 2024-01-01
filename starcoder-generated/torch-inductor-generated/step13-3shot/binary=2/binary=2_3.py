
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x0):
        x1 = torch.transpose(x0, 0, 2)
        x2 = torch.transpose(x1, 0, 1)
        v5 = x2 - 3.76
        return v5
# Inputs to the model
x0 = torch.randn(23, 4, 5)
