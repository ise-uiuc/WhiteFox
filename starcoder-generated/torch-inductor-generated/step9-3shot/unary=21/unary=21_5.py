
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv = torch.nn.Conv2d(12, 12, 3)
        self.conv0 = torch.nn.Conv2d(14, 12, 3)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv0(torch.cat([v1, v2, x2], dim=1))
        return v3
# Inputs to the model
x1 = torch.randn(1, 12, 25, 25)
x2 = torch.randn(1, 14, 25, 25)
