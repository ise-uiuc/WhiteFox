
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3, stride=1, padding=0)
        self.dropout = torch.nn.Dropout(0.1, False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.nn.functional.dropout(v1 + 3, 0.5, True)
        v3 = v1 * v2
        v4 = v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
