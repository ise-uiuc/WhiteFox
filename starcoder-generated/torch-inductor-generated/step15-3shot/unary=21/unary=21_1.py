
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 3, 1)
    def forward(self, x):
        v1 = torch.tanh(self.conv(x))
        return v1
# Inputs to the model
x = torch.randn(12, 6, 128, 128)
