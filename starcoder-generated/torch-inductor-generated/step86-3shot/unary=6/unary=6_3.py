
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout2d(p=0.75)
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.dropout(v1)
        v3 = 3 + v2
        v4 = torch.clamp(v3, 0, 6)
        v5 = v2 * v4
        v6 = v5 / 6
        return v6.unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
