
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg1 = torch.nn.AvgPool2d((1,7), stride=1, padding=1, ceil_mode=True)
        self.avg2 = torch.nn.AvgPool2d((1,2), stride=1, padding=0, ceil_mode=False)
    def forward(self, x1):
        v1 = self.avg1(x1)
        v2 = self.avg2(x1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
