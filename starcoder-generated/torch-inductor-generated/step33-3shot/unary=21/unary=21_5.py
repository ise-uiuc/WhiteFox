
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 15, 5, stride=1, padding=1, dilation=1)
        self.activation = torch.nn.Tanh()
    def forward(self, x15):
        x16 = self.conv(x15)
        x17 = self.activation(x16)
        return x17
# Inputs to the model
x15 = torch.randn(1, 3, 8, 8)
