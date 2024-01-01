
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.dropout = torch.nn.Dropout2d(0.0)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.dropout(v1)
        v2 = self.sigmoid(v2)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
