
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1 = torch.nn.Conv2d(1, 16, (3, 3), groups=2)
        self.BN1 = torch.nn.BatchNorm2d(16)
        self.AvgPool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.ReLU = torch.nn.ReLU()
    def forward(self, x, y):
        x = self.Conv1(x) + y
        x = self.BN1(x)
        x = self.ReLU(x)
        x = self.AvgPool(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 16, 16)
y = torch.randn(1, 1, 16, 16)
