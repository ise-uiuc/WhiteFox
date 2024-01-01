
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x1, x2, alpha=0.5):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        beta = 1-alpha
        v3 = alpha*v1 + beta*v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
