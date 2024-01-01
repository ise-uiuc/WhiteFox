
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = torch.tanh(self.conv(x))
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x = torch.randn(10, 3, 30, 30)
