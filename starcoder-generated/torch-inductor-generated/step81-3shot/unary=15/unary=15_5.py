
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(3, stride=2, padding=1)
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, x1):
        v1 = self.avgpool(x1)
        v2 = self.dropout(v1)
        return v2
# Inputs to the model
x1 = torch.randn(192, 2048, 7, 7)
