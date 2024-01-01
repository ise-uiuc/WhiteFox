
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1, stride=1, padding=0)
        self.dropout = torch.nn.Dropout2d(0.5)
        self.conv2 = torch.nn.Conv2d(32, 12, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.dropout(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(6, 3, 256, 256)
