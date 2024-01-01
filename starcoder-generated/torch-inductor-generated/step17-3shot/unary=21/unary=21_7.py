
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_a = torch.nn.Conv2d(3, 40, 33, stride=5, padding=1, dilation=8)
        self.conv_b = torch.nn.ConvTranspose2d(40, 3, 14)
    def forward(self, x):
        v1 = self.conv_a(x)
        v2 = torch.tanh(v1)
        v3 = self.conv_b(v2)
        return v3
# Inputs to the model
x = torch.randn(2, 3, 57, 50)
