
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        x = torch.sigmoid(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
