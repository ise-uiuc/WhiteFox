
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 8, 1, stride=1, padding=1)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 7, 512, 512)
