
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 3, stride=1,padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        x2 = self.conv(x1)
        return x2
# Inputs to the model
x1 = torch.randn(2, 5, 64, 64)
