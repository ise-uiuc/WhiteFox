
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 2, 1, stride=1, padding=3, dilation=3)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = F.interpolate(x, size=(24, 24))
        v2 = self.conv(v1)
        v3 = F.dropout(v2, training=self.training)
        v4 = self.tanh(v3)
        return v2
# Inputs to the model
x = torch.randn(1, 6, 28, 28)
