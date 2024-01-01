
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 32, 3, stride=2)
    def forward(self, x1):
        x2 = self.conv2d(x1)
        x3 = torch.rand_like(x1)
        return x3
# Inputs to the model
x1 = torch.randn(1, 3, 7, 7)
