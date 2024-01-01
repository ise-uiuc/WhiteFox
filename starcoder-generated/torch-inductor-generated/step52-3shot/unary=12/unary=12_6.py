
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 7, 3, stride=2, padding=1, dilation=1, groups=3)
        self.dropout = torch.nn.Dropout2d(0.391943)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.dropout(v1)
        return v2
# Inputs to the model
x1 = torch.Tensor(1, 8, 64, 64)
