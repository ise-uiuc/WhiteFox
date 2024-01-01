
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 7, 1, stride=1, bias=False)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 64, 196, 196)
