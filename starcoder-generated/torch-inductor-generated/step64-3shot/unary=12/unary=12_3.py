
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 16, 5, stride=2, padding=0)
        self.dropout2d = torch.nn.Dropout2d(p=0.25)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.dropout2d(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 6, 32, 32)
