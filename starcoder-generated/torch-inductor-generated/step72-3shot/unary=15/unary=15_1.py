
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = torch.nn.AvgPool2d(3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.max_pool(x1)
        v2 = self.avg_pool(x1)
        return v1# + v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
