
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(17, 17, 2, stride=1, padding=0)
    def forward(self, x):
        v2 = self.conv(x)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 17, 18, 48)
