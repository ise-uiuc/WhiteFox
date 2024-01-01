
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 5, 1, stride=1, padding=1)
        self.dropout = torch.nn.Dropout(p=0.2)
    def forward(self, x1, other=1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v5 = v2 + other
        v6 = self.dropout(v5)
        v3 = v1 - other
        v4 = v3 + other
        v7 = self.dropout(v4)
        return v7
# Inputs to the model
x1 = torch.randn(1, 10, 64, 64)
